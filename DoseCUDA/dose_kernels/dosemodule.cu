#include <algorithm>
#include <climits>
#include <memory>
#include <vector>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>

#include "IMRTClasses.cuh"
#include "IMPTClasses.cuh"
#include "IMPTWeightGradients.cuh"
#include "IMPTWeightOptimizer.cuh"
#include "IMPTInfluenceMatrix.cuh"
#include "MemoryClasses.h"


/** @brief Check a NumPy array for dimensionality and contained element type
 * 	@param arr
 * 		NumPy array pointer
 * 	@param dim
 * 		Required dimensionality
 * 	@param type
 * 		Required type constant
 * 	@returns true if the constraint is satisfied, false if not
 */
static bool pyarray_typecheck(const PyArrayObject *arr, int dim, int type) {

	return PyArray_NDIM(arr) == dim && PyArray_TYPE(arr) == type;
}


/** @brief Fetch the array's data pointer as a pointer to T */
template <class T>
static T *pyarray_as(PyArrayObject *arr) {

	return reinterpret_cast<T *>(PyArray_DATA(arr));
}


/** @brief Borrow a reference to a NumPy array object contained by a class
 * 	@param self
 * 		Class containing the array
 * 	@param attr
 * 		Member/field name of the array
 * 	@param dim
 * 		Dimensionality/rank of the array
 * 	@param[out] arr
 * 		The `PyArrayObject *` will be written here on success. You do not need
 * 		to `Py_DECREF` this object
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will already have been raised
 */
static bool pyobject_getarray(PyObject *self, const char *attr, int dim, PyArrayObject **arr) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}

	bool result = false;
	*arr = reinterpret_cast<PyArrayObject *>(ptr);
	if (PyArray_Check(ptr) && pyarray_typecheck(*arr, dim, NPY_FLOAT)) {
		result = true;
	} else {
		PyErr_Format(PyExc_ValueError, "'%s' must be %d-dimensional and of type float.", attr, dim);
	}
	Py_DECREF(ptr);
	return result;
}


/** @brief Get a `double` from a Python class
 * 	@param self
 * 		Class
 * 	@param attr
 * 		Member/field name of the float
 * 	@param[out] value
 * 		The result will be written here on success
 * 	@returns true on success, false on error. If an error occurs, a Python
 * 		exception will have been raised
 */
static bool pyobject_getfloat(PyObject *self, const char *attr, double *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	*value = PyFloat_AsDouble(ptr);
	Py_DECREF(ptr);
	return *value == -1.0 ? !PyErr_Occurred() : true;
}


static bool pyobject_getbool(PyObject *self, const char *attr, bool *value) {

	PyObject *ptr = PyObject_GetAttrString(self, attr);
	if (!ptr) {
		return false;
	}
	int result = PyObject_IsTrue(ptr);
	Py_DECREF(ptr);
	*value = result > 0;
	return result >= 0;
}


static PyObject* proton_raytrace(PyObject *self, PyObject *args) {

	PyObject *model_instance, *volume_instance, *beam_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &beam_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double vsadx, vsady;

	if (!pyobject_getfloat(model_instance, "VSADX", &vsadx)
	 || !pyobject_getfloat(model_instance, "VSADY", &vsady)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check beam properties
	PyArrayObject *iso_array;
	if (!pyobject_getarray(beam_instance, "iso", 1, &iso_array)) {
		return NULL;
	}

	double ga, ta;
	if (!pyobject_getfloat(beam_instance, "gantry_angle", &ga)
	 || !pyobject_getfloat(beam_instance, "couch_angle", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	float adjusted_gantry_angle = fmodf(ga + 180.0f, 360.0f);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		// beam model object
		auto model = IMPTBeam::Model();
		model.vsadx = vsadx;
		model.vsady = vsady;

		// beam object
		IMPTBeam beam_obj = IMPTBeam(adjusted_isocenter, adjusted_ga, ta, &model);

		// dose object
		IMPTDose dose_obj = IMPTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();

		//perform raytrace
		proton_raytrace_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_wet = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), WETArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_wet, NPY_ARRAY_OWNDATA);

		return return_wet;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static bool spot_compare(const Spot &a, const Spot &b)
{
	return a.energy_id < b.energy_id;
}


/** Create the authoritative spot array from spot data in any order */
static void make_spot_array(PyArrayObject *spots, HostPointer<Spot> &res)
{
	const size_t count = PyArray_DIM(spots, 0);
	const float *src = pyarray_as<float>(spots);

	for (size_t i = 0; i < count; i++) {
		res[i].x = src[4 * i];
		res[i].y = src[4 * i + 1];
		res[i].mu = src[4 * i + 2];
		res[i].energy_id = static_cast<int>(src[4 * i + 3]);
		// printf("Spot %d: x: %f, y: %f, mu: %f, energy_id: %d\n", i, res[i].x, res[i].y, res[i].mu, res[i].energy_id);
	}

	std::sort(&res[0], &res[count], spot_compare);
}


/** Create a sorted spot array while retaining the caller's original order. */
static void make_indexed_spot_array(
	PyArrayObject *spots,
	HostPointer<Spot> &res,
	HostPointer<size_t> &sorted_to_original)
{
	struct IndexedSpot {
		Spot spot;
		size_t original_index;
	};

	const size_t count = PyArray_DIM(spots, 0);
	const float *src = pyarray_as<float>(spots);
	std::vector<IndexedSpot> indexed(count);

	for (size_t i = 0; i < count; ++i) {
		indexed[i].spot.x = src[4 * i];
		indexed[i].spot.y = src[4 * i + 1];
		indexed[i].spot.mu = src[4 * i + 2];
		indexed[i].spot.energy_id = static_cast<int>(src[4 * i + 3]);
		indexed[i].original_index = i;
	}

	std::sort(indexed.begin(), indexed.end(), [](const IndexedSpot &a, const IndexedSpot &b) {
		return a.spot.energy_id < b.spot.energy_id;
	});

	for (size_t i = 0; i < count; ++i) {
		res[i] = indexed[i].spot;
		sorted_to_original[i] = indexed[i].original_index;
	}
}


static void make_mlc_array(PyArrayObject *mlc, HostPointer<MLCPair> &res)
{
	const size_t count = PyArray_DIM(mlc, 0);
	const float *src = pyarray_as<float>(mlc);

	for (size_t i = 0; i < count; i++) {
		res[i].x1 = src[i];
		res[i].x2 = src[i + count];
		res[i].y_offset = src[i + 2 * count];
		res[i].y_width = src[i + 3 * count];
		// printf("MLC Pair %d: x1: %f, x2: %f, y_offset: %f, y_width: %f\n", i, res[i].x1, res[i].x2, res[i].y_offset, res[i].y_width);
	}
}


static PyObject* proton_spot(PyObject *self, PyObject *args) {

	PyObject *model_instance, *volume_instance, *wet_instance, *beam_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOOi", &model_instance, &volume_instance, &wet_instance, &beam_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double vsadx, vsady;

	if (!pyobject_getfloat(model_instance, "VSADX", &vsadx)
	 || !pyobject_getfloat(model_instance, "VSADY", &vsady)) {
		return NULL;
	}

	PyArrayObject *lut_depths_array, *lut_sigmas_array, *lut_idds_array, *lut_divergence_params_array;
	if (!pyobject_getarray(model_instance, "divergence_params", 2, &lut_divergence_params_array)
	|| !pyobject_getarray(model_instance, "lut_depths", 2, &lut_depths_array)
	|| !pyobject_getarray(model_instance, "lut_sigmas", 2, &lut_sigmas_array)
	|| !pyobject_getarray(model_instance, "lut_idds", 2, &lut_idds_array)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check WET data properties
	PyArrayObject *wet_array;
	if (!pyobject_getarray(wet_instance, "voxel_data", 3, &wet_array)) {
		return NULL;
	}

	// check beam properties
	PyArrayObject *iso_array, *spots_array;
	if (!pyobject_getarray(beam_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(beam_instance, "spot_list", 2, &spots_array)) {
		return NULL;
	}

	double ga, ta;
	if (!pyobject_getfloat(beam_instance, "gantry_angle", &ga)
	 || !pyobject_getfloat(beam_instance, "couch_angle", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);
		size_t n_energies = PyArray_DIM(lut_depths_array, 0);
		size_t n_spots = PyArray_DIM(spots_array, 0);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(wet_array)[0],
			(size_t)PyArray_DIMS(wet_array)[1],
			(size_t)PyArray_DIMS(wet_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		IMPTDose dose_obj = IMPTDose(dims, voxel_sp);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DoseArray = DoseArray.get();
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = pyarray_as<float>(wet_array);

		// beam model object
		auto model = IMPTBeam::Model();
		model.vsadx = vsadx;
		model.vsady = vsady;

		// beam object
		IMPTBeam beam_obj = IMPTBeam(adjusted_isocenter, adjusted_ga, ta, &model);

		HostPointer<Layer> LayerArray(n_energies);
		HostPointer<Spot> SpotArray(n_spots);

		make_spot_array(spots_array, SpotArray);

		beam_obj.n_energies = n_energies;
		beam_obj.layers = LayerArray.get();
		beam_obj.spots = SpotArray.get();
		beam_obj.n_spots = n_spots;
		beam_obj.divergence_params = pyarray_as<float>(lut_divergence_params_array);
		beam_obj.dvp_len = 5;	// R80, energy, quadratic coefficients
		beam_obj.lut_depths = pyarray_as<float>(lut_depths_array);
		beam_obj.lut_sigmas = pyarray_as<float>(lut_sigmas_array);
		beam_obj.lut_idds = pyarray_as<float>(lut_idds_array);
		beam_obj.lut_len = LUT_LENGTH;	// This can now be changed at runtime

		beam_obj.importLayers();

		//compute dose
		proton_spot_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(wet_array), PyArray_TYPE(wet_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


/** Compute dL/dMU from an arbitrary voxel adjoint dL/dDose. */
static PyObject* proton_spot_weight_vjp(PyObject *self, PyObject *args) {

	PyObject *model_instance, *volume_instance, *wet_instance, *beam_instance;
	PyObject *dose_adjoint_instance;
	int gpu_id;

	if (!PyArg_ParseTuple(
		args,
		"OOOOOi",
		&model_instance,
		&volume_instance,
		&wet_instance,
		&beam_instance,
		&dose_adjoint_instance,
		&gpu_id)) {
		return NULL;
	}

	double vsadx, vsady;
	if (!pyobject_getfloat(model_instance, "VSADX", &vsadx)
	 || !pyobject_getfloat(model_instance, "VSADY", &vsady)) {
		return NULL;
	}

	PyArrayObject *lut_depths_array, *lut_sigmas_array, *lut_idds_array;
	PyArrayObject *lut_divergence_params_array;
	if (!pyobject_getarray(model_instance, "divergence_params", 2, &lut_divergence_params_array)
	 || !pyobject_getarray(model_instance, "lut_depths", 2, &lut_depths_array)
	 || !pyobject_getarray(model_instance, "lut_sigmas", 2, &lut_sigmas_array)
	 || !pyobject_getarray(model_instance, "lut_idds", 2, &lut_idds_array)) {
		return NULL;
	}

	PyArrayObject *density_array, *spacing_array, *origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	PyArrayObject *wet_array;
	if (!pyobject_getarray(wet_instance, "voxel_data", 3, &wet_array)) {
		return NULL;
	}

	if (!PyArray_Check(dose_adjoint_instance)) {
		PyErr_SetString(PyExc_ValueError, "dose_adjoint must be a NumPy array");
		return NULL;
	}
	PyArrayObject *dose_adjoint_array =
		reinterpret_cast<PyArrayObject *>(dose_adjoint_instance);
	if (!pyarray_typecheck(dose_adjoint_array, 3, NPY_FLOAT)
	 || !PyArray_IS_C_CONTIGUOUS(dose_adjoint_array)) {
		PyErr_SetString(
			PyExc_ValueError,
			"dose_adjoint must be a C-contiguous, 3-dimensional float32 array");
		return NULL;
	}
	for (int axis = 0; axis < 3; ++axis) {
		if (PyArray_DIM(dose_adjoint_array, axis) != PyArray_DIM(wet_array, axis)) {
			PyErr_SetString(PyExc_ValueError, "dose_adjoint shape must match the WET volume");
			return NULL;
		}
	}

	PyArrayObject *iso_array, *spots_array;
	if (!pyobject_getarray(beam_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(beam_instance, "spot_list", 2, &spots_array)) {
		return NULL;
	}

	double ga, ta;
	if (!pyobject_getfloat(beam_instance, "gantry_angle", &ga)
	 || !pyobject_getfloat(beam_instance, "couch_angle", &ta)) {
		return NULL;
	}

	const size_t n_spots = PyArray_DIM(spots_array, 0);
	if (n_spots == 0) {
		PyErr_SetString(PyExc_ValueError, "beam must contain at least one spot");
		return NULL;
	}

	float *spacing = pyarray_as<float>(spacing_array);
	float *origin = pyarray_as<float>(origin_array);
	float *iso = pyarray_as<float>(iso_array);

	try {
		const float adjusted_ga = fmodf(ga + 180.0f, 360.0f);
		const size_t n_energies = PyArray_DIM(lut_depths_array, 0);
		size_t dims[3] = {
			static_cast<size_t>(PyArray_DIMS(wet_array)[0]),
			static_cast<size_t>(PyArray_DIMS(wet_array)[1]),
			static_cast<size_t>(PyArray_DIMS(wet_array)[2]),
		};
		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2],
		};

		IMPTDose dose_obj = IMPTDose(dims, spacing[0]);
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = pyarray_as<float>(wet_array);

		auto model = IMPTBeam::Model();
		model.vsadx = vsadx;
		model.vsady = vsady;
		IMPTBeam beam_obj = IMPTBeam(adjusted_isocenter, adjusted_ga, ta, &model);

		HostPointer<Layer> LayerArray(n_energies);
		HostPointer<Spot> SpotArray(n_spots);
		HostPointer<size_t> SortedToOriginal(n_spots);
		make_indexed_spot_array(spots_array, SpotArray, SortedToOriginal);

		beam_obj.n_energies = n_energies;
		beam_obj.layers = LayerArray.get();
		beam_obj.spots = SpotArray.get();
		beam_obj.n_spots = n_spots;
		beam_obj.divergence_params = pyarray_as<float>(lut_divergence_params_array);
		beam_obj.dvp_len = 5;
		beam_obj.lut_depths = pyarray_as<float>(lut_depths_array);
		beam_obj.lut_sigmas = pyarray_as<float>(lut_sigmas_array);
		beam_obj.lut_idds = pyarray_as<float>(lut_idds_array);
		beam_obj.lut_len = LUT_LENGTH;
		beam_obj.importLayers();

		HostPointer<float> SortedGradient(MemoryTag::Zeroed(), n_spots);
		HostPointer<float> OriginalGradient(MemoryTag::Zeroed(), n_spots);
		proton_spot_weight_vjp_cuda(
			gpu_id,
			&dose_obj,
			&beam_obj,
			pyarray_as<float>(dose_adjoint_array),
			SortedGradient.get());

		for (size_t sorted_index = 0; sorted_index < n_spots; ++sorted_index) {
			OriginalGradient[SortedToOriginal[sorted_index]] = SortedGradient[sorted_index];
		}

		npy_intp gradient_shape[1] = { static_cast<npy_intp>(n_spots) };
		PyObject *return_gradient = PyArray_SimpleNewFromData(
			1,
			gradient_shape,
			NPY_FLOAT,
			OriginalGradient.release());
		PyArray_ENABLEFLAGS(
			reinterpret_cast<PyArrayObject *>(return_gradient),
			NPY_ARRAY_OWNDATA);
		return return_gradient;

	} catch (std::bad_alloc &) {
		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");
	} catch (std::runtime_error &e) {
		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());
	}

	return NULL;
}


struct PyObjectDecref {
	void operator()(PyObject *object) const { Py_XDECREF(object); }
};

/** Host-side geometry for the experimental, separate CUDA weight solver. */
struct PreparedWeightBeam {
	std::unique_ptr<HostPointer<Layer>> layers;
	std::unique_ptr<HostPointer<Spot>> spots;
	std::unique_ptr<HostPointer<size_t>> sorted_to_original;
	std::unique_ptr<IMPTBeam> beam;
	std::unique_ptr<IMPTDose> dose;
	int original_offset;

	PreparedWeightBeam(PyObject *model_object, PyObject *wet_object,
	                   PyObject *beam_object, PyArrayObject *volume,
	                   PyArrayObject *spacing, PyArrayObject *origin,
	                   int offset) : original_offset(offset) {
		double vsadx, vsady, ga, ta;
		if (!pyobject_getfloat(model_object, "VSADX", &vsadx)
		 || !pyobject_getfloat(model_object, "VSADY", &vsady)
		 || !pyobject_getfloat(beam_object, "gantry_angle", &ga)
		 || !pyobject_getfloat(beam_object, "couch_angle", &ta)) {
			throw std::invalid_argument("invalid beam geometry attributes");
		}
		PyArrayObject *divergence, *depths, *sigmas, *idds, *wet;
		PyArrayObject *iso, *spot_array;
		if (!pyobject_getarray(model_object, "divergence_params", 2, &divergence)
		 || !pyobject_getarray(model_object, "lut_depths", 2, &depths)
		 || !pyobject_getarray(model_object, "lut_sigmas", 2, &sigmas)
		 || !pyobject_getarray(model_object, "lut_idds", 2, &idds)
		 || !pyobject_getarray(wet_object, "voxel_data", 3, &wet)
		 || !pyobject_getarray(beam_object, "iso", 1, &iso)
		 || !pyobject_getarray(beam_object, "spot_list", 2, &spot_array)) {
			throw std::invalid_argument("invalid beam arrays");
		}
		for (int axis = 0; axis < 3; ++axis) {
			if (PyArray_DIM(wet, axis) != PyArray_DIM(volume, axis)) {
				throw std::invalid_argument("WET shape differs from dose volume");
			}
		}
		if (PyArray_DIM(spot_array, 1) != 4 || PyArray_DIM(spot_array, 0) <= 0) {
			throw std::invalid_argument("spot_list must have shape (n, 4)");
		}
		if (!PyArray_IS_C_CONTIGUOUS(divergence)
		 || !PyArray_IS_C_CONTIGUOUS(depths)
		 || !PyArray_IS_C_CONTIGUOUS(sigmas)
		 || !PyArray_IS_C_CONTIGUOUS(idds)
		 || !PyArray_IS_C_CONTIGUOUS(wet)
		 || !PyArray_IS_C_CONTIGUOUS(iso)
		 || !PyArray_IS_C_CONTIGUOUS(spot_array)
		 || PyArray_DIM(iso, 0) != 3) {
			throw std::invalid_argument("beam arrays must be contiguous with a 3-vector isocenter");
		}
		const int n_energies = static_cast<int>(PyArray_DIM(depths, 0));
		const int n_spots = static_cast<int>(PyArray_DIM(spot_array, 0));
		if (!n_energies || PyArray_DIM(divergence, 0) != n_energies
		    || PyArray_DIM(sigmas, 0) != n_energies
		    || PyArray_DIM(idds, 0) != n_energies
		    || PyArray_DIM(divergence, 1) != 5
		    || PyArray_DIM(depths, 1) != LUT_LENGTH
		    || PyArray_DIM(sigmas, 1) != LUT_LENGTH
		    || PyArray_DIM(idds, 1) != LUT_LENGTH) {
			throw std::invalid_argument("beam-model energy dimensions disagree");
		}
		size_t dims[3] = {
			static_cast<size_t>(PyArray_DIM(volume, 0)),
			static_cast<size_t>(PyArray_DIM(volume, 1)),
			static_cast<size_t>(PyArray_DIM(volume, 2))};
		float *origin_data = pyarray_as<float>(origin);
		float *iso_data = pyarray_as<float>(iso);
		float adjusted_iso[3] = {
			iso_data[0] - origin_data[0],
			iso_data[1] - origin_data[1],
			iso_data[2] - origin_data[2]};
		IMPTBeam::Model model;
		model.vsadx = static_cast<float>(vsadx);
		model.vsady = static_cast<float>(vsady);
		beam.reset(new IMPTBeam(adjusted_iso,
			fmodf(static_cast<float>(ga) + 180.0f, 360.0f),
			static_cast<float>(ta), &model));
		layers.reset(new HostPointer<Layer>(n_energies));
		spots.reset(new HostPointer<Spot>(n_spots));
		sorted_to_original.reset(new HostPointer<size_t>(n_spots));
		make_indexed_spot_array(spot_array, *spots, *sorted_to_original);
		beam->n_energies = n_energies;
		beam->layers = layers->get();
		beam->spots = spots->get();
		beam->n_spots = n_spots;
		beam->divergence_params = pyarray_as<float>(divergence);
		beam->dvp_len = 5;
		beam->lut_depths = pyarray_as<float>(depths);
		beam->lut_sigmas = pyarray_as<float>(sigmas);
		beam->lut_idds = pyarray_as<float>(idds);
		beam->lut_len = LUT_LENGTH;
		beam->importLayers();
		dose.reset(new IMPTDose(dims, pyarray_as<float>(spacing)[0]));
		dose->DensityArray = pyarray_as<float>(volume);
		dose->WETArray = pyarray_as<float>(wet);
	}
};


static PyObject* proton_optimize_spot_weights(PyObject *self, PyObject *args) {
	PyObject *models_object, *volume_object, *wets_object, *beams_object;
	PyObject *masks_object, *weights_object;
	double prescription, oar_limit, normal_limit, dose_scale, tolerance;
	int max_iterations, gpu_id;
	const char *method;
	if (!PyArg_ParseTuple(args, "OOOOOddddOidsi", &models_object,
	    &volume_object, &wets_object, &beams_object, &masks_object,
	    &prescription, &oar_limit, &normal_limit, &dose_scale,
	    &weights_object, &max_iterations, &tolerance, &method, &gpu_id)) {
		return NULL;
	}
	std::unique_ptr<PyObject, PyObjectDecref> models(
		PySequence_Fast(models_object, "models must be a sequence"));
	std::unique_ptr<PyObject, PyObjectDecref> wets(
		PySequence_Fast(wets_object, "WET volumes must be a sequence"));
	std::unique_ptr<PyObject, PyObjectDecref> beams(
		PySequence_Fast(beams_object, "beams must be a sequence"));
	std::unique_ptr<PyObject, PyObjectDecref> masks(
		PySequence_Fast(masks_object, "masks must be a sequence"));
	if (!models || !wets || !beams || !masks) return NULL;
	const Py_ssize_t beam_count = PySequence_Fast_GET_SIZE(beams.get());
	if (beam_count == 0 || PySequence_Fast_GET_SIZE(models.get()) != beam_count
	    || PySequence_Fast_GET_SIZE(wets.get()) != beam_count
	    || PySequence_Fast_GET_SIZE(masks.get()) != 3) {
		PyErr_SetString(PyExc_ValueError, "beam/model/WET counts or mask count differ");
		return NULL;
	}
	PyArrayObject *volume, *spacing, *origin;
	if (!pyobject_getarray(volume_object, "voxel_data", 3, &volume)
	 || !pyobject_getarray(volume_object, "spacing", 1, &spacing)
	 || !pyobject_getarray(volume_object, "origin", 1, &origin)) return NULL;
	if (!PyArray_IS_C_CONTIGUOUS(volume)
	    || !PyArray_IS_C_CONTIGUOUS(spacing)
	    || !PyArray_IS_C_CONTIGUOUS(origin)
	    || PyArray_DIM(spacing, 0) != 3 || PyArray_DIM(origin, 0) != 3) {
		PyErr_SetString(PyExc_ValueError, "invalid dose volume layout");
		return NULL;
	}
	PyArrayObject *mask_arrays[3];
	for (int i = 0; i < 3; ++i) {
		PyObject *item = PySequence_Fast_GET_ITEM(masks.get(), i);
		if (!PyArray_Check(item)) {
			PyErr_SetString(PyExc_ValueError, "masks must be NumPy arrays");
			return NULL;
		}
		mask_arrays[i] = reinterpret_cast<PyArrayObject *>(item);
		if (!pyarray_typecheck(mask_arrays[i], 3, NPY_UBYTE)
		    || !PyArray_IS_C_CONTIGUOUS(mask_arrays[i])) {
			PyErr_SetString(PyExc_ValueError, "masks must be contiguous uint8 volumes");
			return NULL;
		}
		for (int axis = 0; axis < 3; ++axis) {
			if (PyArray_DIM(mask_arrays[i], axis) != PyArray_DIM(volume, axis)) {
				PyErr_SetString(PyExc_ValueError, "mask shape differs from dose volume");
				return NULL;
			}
		}
	}
	if (!PyArray_Check(weights_object)) {
		PyErr_SetString(PyExc_ValueError, "weights must be a NumPy array");
		return NULL;
	}
	PyArrayObject *weights_array = reinterpret_cast<PyArrayObject *>(weights_object);
	if (!pyarray_typecheck(weights_array, 1, NPY_FLOAT)
	    || !PyArray_IS_C_CONTIGUOUS(weights_array)) {
		PyErr_SetString(PyExc_ValueError, "weights must be contiguous float32");
		return NULL;
	}
	try {
		std::vector<std::unique_ptr<PreparedWeightBeam>> prepared;
		std::vector<IMPTBeam *> host_beams;
		std::vector<IMPTDose *> host_doses;
		int total_spots = 0;
		for (Py_ssize_t i = 0; i < beam_count; ++i) {
			prepared.emplace_back(new PreparedWeightBeam(
				PySequence_Fast_GET_ITEM(models.get(), i),
				PySequence_Fast_GET_ITEM(wets.get(), i),
				PySequence_Fast_GET_ITEM(beams.get(), i), volume, spacing, origin,
				total_spots));
			host_beams.push_back(prepared.back()->beam.get());
			host_doses.push_back(prepared.back()->dose.get());
			total_spots += prepared.back()->beam->n_spots;
		}
		if (PyArray_DIM(weights_array, 0) != total_spots) {
			PyErr_SetString(PyExc_ValueError, "initial weight count differs from spots");
			return NULL;
		}
		std::vector<float> sorted_weights(total_spots);
		const float *original = pyarray_as<float>(weights_array);
		for (const auto &entry : prepared) {
			for (int i = 0; i < entry->beam->n_spots; ++i) {
				sorted_weights[entry->original_offset + i] =
					original[entry->original_offset + (*entry->sorted_to_original)[i]];
			}
		}
		const auto result = optimize_impt_weights_cuda(
			gpu_id, host_doses, host_beams,
			pyarray_as<unsigned char>(mask_arrays[0]),
			pyarray_as<unsigned char>(mask_arrays[1]),
			pyarray_as<unsigned char>(mask_arrays[2]),
			static_cast<float>(prescription), static_cast<float>(oar_limit),
			static_cast<float>(normal_limit), static_cast<float>(dose_scale),
			sorted_weights.data(), max_iterations,
			static_cast<float>(tolerance), method);
		npy_intp shape[1] = { total_spots };
		PyObject *weights_result = PyArray_SimpleNew(1, shape, NPY_FLOAT);
		if (!weights_result) return NULL;
		float *output = pyarray_as<float>(
			reinterpret_cast<PyArrayObject *>(weights_result));
		for (const auto &entry : prepared) {
			for (int i = 0; i < entry->beam->n_spots; ++i) {
				output[entry->original_offset + (*entry->sorted_to_original)[i]] =
					result.weights[entry->original_offset + i];
			}
		}
		return Py_BuildValue("{s:N,s:f,s:f,s:i,s:i,s:i,s:O}",
			"weights", weights_result,
			"objective", result.objective,
			"projected_gradient", result.projected_gradient,
			"iterations", result.iterations,
			"forward_evaluations", result.forward_evaluations,
			"gradient_evaluations", result.gradient_evaluations,
			"converged", result.converged ? Py_True : Py_False);
	} catch (std::bad_alloc &) {
		PyErr_SetString(PyExc_MemoryError, "not enough memory for CUDA weight solver");
	} catch (std::invalid_argument &error) {
		if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
	} catch (std::runtime_error &error) {
		PyErr_Format(PyExc_RuntimeError, "CUDA weight solver: %s", error.what());
	}
	return NULL;
}


static const char *GPU_MATRIX_CAPSULE_NAME = "DoseCUDA.GPUInfluenceMatrix";

static void gpu_matrix_capsule_destroy(PyObject *capsule) {
	void *pointer = PyCapsule_GetPointer(capsule, GPU_MATRIX_CAPSULE_NAME);
	if (pointer) delete static_cast<GPUInfluenceMatrix *>(pointer);
	else PyErr_Clear();
}


static PyObject *proton_gpu_matrix_create(PyObject *self, PyObject *args) {
	PyObject *matrix_object, *target_object, *oar_object, *normal_object;
	double prescription, oar_limit, normal_limit, oar_weight, normal_weight;
	int gpu_id;
	if (!PyArg_ParseTuple(args, "OOOOdddddi", &matrix_object, &target_object,
	    &oar_object, &normal_object, &prescription, &oar_limit, &normal_limit,
	    &oar_weight, &normal_weight, &gpu_id)) return NULL;
	if (!PyArray_Check(matrix_object)) {
		PyErr_SetString(PyExc_TypeError, "matrix must be a NumPy array");
		return NULL;
	}
	auto *matrix = reinterpret_cast<PyArrayObject *>(matrix_object);
	if (!pyarray_typecheck(matrix, 2, NPY_DOUBLE)
	    || !PyArray_IS_C_CONTIGUOUS(matrix)
	    || PyArray_DIM(matrix, 0) <= 0 || PyArray_DIM(matrix, 1) <= 0
	    || PyArray_DIM(matrix, 0) > INT_MAX || PyArray_DIM(matrix, 1) > INT_MAX) {
		PyErr_SetString(PyExc_ValueError,
			"matrix must be nonempty, contiguous float64 (voxels, spots)");
		return NULL;
	}
	PyObject *mask_objects[3] = {target_object, oar_object, normal_object};
	PyArrayObject *masks[3];
	for (int i = 0; i < 3; ++i) {
		if (!PyArray_Check(mask_objects[i])) {
			PyErr_SetString(PyExc_TypeError, "masks must be NumPy arrays");
			return NULL;
		}
		masks[i] = reinterpret_cast<PyArrayObject *>(mask_objects[i]);
		if (!pyarray_typecheck(masks[i], 1, NPY_UBYTE)
		    || !PyArray_IS_C_CONTIGUOUS(masks[i])
		    || PyArray_DIM(masks[i], 0) != PyArray_DIM(matrix, 0)) {
			PyErr_SetString(PyExc_ValueError,
				"masks must be contiguous uint8 vectors matching matrix voxels");
			return NULL;
		}
	}
	try {
		std::unique_ptr<GPUInfluenceMatrix> context(new GPUInfluenceMatrix(
			gpu_id, pyarray_as<double>(matrix),
			static_cast<int>(PyArray_DIM(matrix, 0)),
			static_cast<int>(PyArray_DIM(matrix, 1)),
			pyarray_as<unsigned char>(masks[0]),
			pyarray_as<unsigned char>(masks[1]),
			pyarray_as<unsigned char>(masks[2]), prescription, oar_limit,
			normal_limit, oar_weight, normal_weight));
		PyObject *capsule = PyCapsule_New(context.get(),
			GPU_MATRIX_CAPSULE_NAME, gpu_matrix_capsule_destroy);
		if (capsule) context.release();
		return capsule;
	} catch (std::bad_alloc &) {
		PyErr_SetString(PyExc_MemoryError, "GPU matrix allocation failed");
	} catch (std::invalid_argument &error) {
		PyErr_SetString(PyExc_ValueError, error.what());
	} catch (std::runtime_error &error) {
		PyErr_Format(PyExc_RuntimeError, "GPU matrix: %s", error.what());
	}
	return NULL;
}


static PyObject *proton_gpu_matrix_evaluate(PyObject *self, PyObject *args) {
	PyObject *capsule, *weights_object;
	int return_dose = 0;
	if (!PyArg_ParseTuple(args, "OO|p", &capsule, &weights_object,
	    &return_dose)) return NULL;
	auto *context = static_cast<GPUInfluenceMatrix *>(
		PyCapsule_GetPointer(capsule, GPU_MATRIX_CAPSULE_NAME));
	if (!context) return NULL;
	if (!PyArray_Check(weights_object)) {
		PyErr_SetString(PyExc_TypeError, "weights must be a NumPy array");
		return NULL;
	}
	auto *weights = reinterpret_cast<PyArrayObject *>(weights_object);
	if (!pyarray_typecheck(weights, 1, NPY_DOUBLE)
	    || !PyArray_IS_C_CONTIGUOUS(weights)
	    || PyArray_DIM(weights, 0) != context->spot_count()) {
		PyErr_SetString(PyExc_ValueError,
			"weights must be contiguous float64 matching matrix spots");
		return NULL;
	}
	npy_intp gradient_shape[1] = {context->spot_count()};
	PyObject *gradient = PyArray_SimpleNew(1, gradient_shape, NPY_DOUBLE);
	if (!gradient) return NULL;
	PyObject *dose = NULL;
	if (return_dose) {
		npy_intp dose_shape[1] = {context->voxel_count()};
		dose = PyArray_SimpleNew(1, dose_shape, NPY_DOUBLE);
		if (!dose) { Py_DECREF(gradient); return NULL; }
	}
	try {
		const double objective = context->value_and_gradient(
			pyarray_as<double>(weights),
			pyarray_as<double>(reinterpret_cast<PyArrayObject *>(gradient)),
			dose ? pyarray_as<double>(reinterpret_cast<PyArrayObject *>(dose)) : NULL);
		if (dose) return Py_BuildValue("{s:d,s:N,s:N}", "objective", objective,
			"gradient", gradient, "dose", dose);
		return Py_BuildValue("{s:d,s:N}", "objective", objective,
			"gradient", gradient);
	} catch (std::runtime_error &error) {
		PyErr_Format(PyExc_RuntimeError, "GPU matrix evaluation: %s", error.what());
	}
	Py_DECREF(gradient);
	Py_XDECREF(dose);
	return NULL;
}


static PyObject * photon_dose(PyObject* self, PyObject* args) {

	PyObject *model_instance, *volume_instance, *cp_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &cp_instance, &gpu_id))
        return NULL;

	// check beam model properties 
	double mu_cal, 
		primary_source_distance, 
		scatter_source_distance, 
		primary_source_size, 
		scatter_source_size, 
		mlc_distance, 
		scatter_source_weight, 
		electron_attenuation, 
		electron_src_weight,
		electron_fitted_dmax,
		jaw_transmission,
		mlc_transmission;

	if (!pyobject_getfloat(model_instance, "mu_calibration", &mu_cal)
	 || !pyobject_getfloat(model_instance, "primary_source_distance", &primary_source_distance)
	 || !pyobject_getfloat(model_instance, "scatter_source_distance", &scatter_source_distance)
	 || !pyobject_getfloat(model_instance, "primary_source_size", &primary_source_size)
	 || !pyobject_getfloat(model_instance, "scatter_source_size", &scatter_source_size)
	 || !pyobject_getfloat(model_instance, "mlc_distance", &mlc_distance)
	 || !pyobject_getfloat(model_instance, "scatter_source_weight", &scatter_source_weight)
	 || !pyobject_getfloat(model_instance, "electron_attenuation", &electron_attenuation)
	 || !pyobject_getfloat(model_instance, "electron_source_weight", &electron_src_weight)
	 || !pyobject_getfloat(model_instance, "electron_fitted_dmax", &electron_fitted_dmax)
	 || !pyobject_getfloat(model_instance, "jaw_transmission", &jaw_transmission)
	 || !pyobject_getfloat(model_instance, "mlc_transmission", &mlc_transmission)) {
		return NULL;
	}

	bool has_xjaws, 
		has_yjaws;
	if (!pyobject_getbool(model_instance, "has_xjaws", &has_xjaws)
	 || !pyobject_getbool(model_instance, "has_yjaws", &has_yjaws)) {
		return NULL;
	}

	PyArrayObject *profile_radius, 
		*profile_intensities, 
		*profile_softening, 
		*spectrum_attenuation_coefficients, 
		*spectrum_primary_weights, 
		*spectrum_scatter_weights, 
		*kernel;
	if (!pyobject_getarray(model_instance, "profile_radius", 1, &profile_radius)
	 || !pyobject_getarray(model_instance, "profile_intensities", 1, &profile_intensities)
	 || !pyobject_getarray(model_instance, "profile_softening", 1, &profile_softening)
	 || !pyobject_getarray(model_instance, "spectrum_attenuation_coefficients", 1, &spectrum_attenuation_coefficients)
	 || !pyobject_getarray(model_instance, "spectrum_primary_weights", 1, &spectrum_primary_weights)
	 || !pyobject_getarray(model_instance, "spectrum_scatter_weights", 1, &spectrum_scatter_weights)
	 || !pyobject_getarray(model_instance, "kernel", 2, &kernel)) {
		return NULL;
	}

	// check volume data properties
	PyArrayObject *density_array, 
		*spacing_array, 
		*origin_array;
	if (!pyobject_getarray(volume_instance, "voxel_data", 3, &density_array)
	 || !pyobject_getarray(volume_instance, "spacing", 1, &spacing_array)
	 || !pyobject_getarray(volume_instance, "origin", 1, &origin_array)) {
		return NULL;
	}

	// check control point data properties
	PyArrayObject *iso_array, 
		*mlc_array;
	if (!pyobject_getarray(cp_instance, "iso", 1, &iso_array)
	 || !pyobject_getarray(cp_instance, "mlc", 2, &mlc_array)) {
		return NULL;
	}

	double mu, 
		ga, 
		ca, 
		ta;
	if (!pyobject_getfloat(cp_instance, "mu", &mu)
	 || !pyobject_getfloat(cp_instance, "ga", &ga)
	 || !pyobject_getfloat(cp_instance, "ca", &ca)
	 || !pyobject_getfloat(cp_instance, "ta", &ta)) {
		return NULL;
	}

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	size_t n_mlc_pairs = PyArray_DIM(mlc_array, 0);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		// beam model object
		auto model = IMRTBeam::Model();
		model.n_profile_points = PyArray_DIM(profile_radius, 0);
		model.profile_radius = pyarray_as<float>(profile_radius);
		model.profile_intensities = pyarray_as<float>(profile_intensities);
		model.profile_softening = pyarray_as<float>(profile_softening);
		model.n_spectral_energies = PyArray_DIM(spectrum_attenuation_coefficients, 0);
		model.spectrum_attenuation_coefficients = pyarray_as<float>(spectrum_attenuation_coefficients);
		model.spectrum_primary_weights = pyarray_as<float>(spectrum_primary_weights);
		model.spectrum_scatter_weights = pyarray_as<float>(spectrum_scatter_weights);
		model.mu_cal = mu_cal;
		model.primary_src_dist = primary_source_distance;
		model.scatter_src_dist = scatter_source_distance;
		model.primary_src_size = primary_source_size;
		model.scatter_src_size = scatter_source_size;
		model.mlc_distance = mlc_distance;
		model.scatter_src_weight = scatter_source_weight;
		model.electron_attenuation = electron_attenuation;
		model.electron_src_weight = electron_src_weight;
		model.kernel = pyarray_as<float>(kernel);
		model.has_xjaws = has_xjaws;
		model.has_yjaws = has_yjaws;
		model.electron_fitted_dmax = electron_fitted_dmax;
		model.jaw_transmission = jaw_transmission;
		model.mlc_transmission = mlc_transmission;

		// dose object
		IMRTDose dose_obj = IMRTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();
		dose_obj.DoseArray = DoseArray.get();

		// MLC object_array
		HostPointer<MLCPair> MLCPairArray(n_mlc_pairs);
		make_mlc_array(mlc_array, MLCPairArray);

		// beam object
		IMRTBeam beam_obj = IMRTBeam(adjusted_isocenter, adjusted_ga, ta, ca, &model);
		beam_obj.n_mlc_pairs = n_mlc_pairs;
		beam_obj.mlc = MLCPairArray.get();
		beam_obj.mu = mu;

		// compute dose
    	photon_dose_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static PyMethodDef DoseMethods[] = {
	{
		"proton_raytrace_cuda",
		proton_raytrace,
		METH_VARARGS,
		"Compute WET array for proton dose calc."
	},
	{
		"proton_spot_cuda",
		proton_spot,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{
		"proton_spot_weight_vjp_cuda",
		proton_spot_weight_vjp,
		METH_VARARGS,
		"Apply the fixed-geometry proton spot-dose transpose to a voxel adjoint."
	},
	{
		"proton_optimize_spot_weights_cuda",
		proton_optimize_spot_weights,
		METH_VARARGS,
		"Experimental matrix-free float32 GPU spot-weight solve."
	},
	{
		"proton_gpu_matrix_create",
		proton_gpu_matrix_create,
		METH_VARARGS,
		"Upload fixed-angle dose columns for GPU-resident matrix products."
	},
	{
		"proton_gpu_matrix_evaluate",
		proton_gpu_matrix_evaluate,
		METH_VARARGS,
		"Compute dose objective and weight gradient using a resident GPU matrix."
	},
	{
		"photon_dose_cuda",
		photon_dose,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{ 0 }
};


static struct PyModuleDef dosemodule = {
	PyModuleDef_HEAD_INIT,
	"dose_kernels",
	"Compute dose on the GPU.",
	-1,
	DoseMethods,
};


PyMODINIT_FUNC PyInit_dose_kernels(void) {
	import_array();
	return PyModule_Create(&dosemodule);
}
