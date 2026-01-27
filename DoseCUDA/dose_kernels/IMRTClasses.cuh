#ifndef IMRTCLASSES_H
#define IMRTCLASSES_H

#include "CudaClasses.cuh"

#define AIR_DENSITY 0.0012f // g/cc

typedef struct {

    float x1;
    float x2;
    float y_offset;
    float y_width;

} MLCPair;

__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n);

class IMRTBeam : public CudaBeam{

    public:

        struct Model {

            int n_profile_points;
            float * profile_radius;
            float * profile_intensities;
            float * profile_softening;

            int n_spectral_energies;
            float * spectrum_attenuation_coefficients;
            float * spectrum_primary_weights;
            float * spectrum_scatter_weights;

            float mu_cal;

            float primary_src_dist;
            float scatter_src_dist;

            float primary_src_size;
            float scatter_src_size;

            float mlc_distance;

            float scatter_src_weight;
            float electron_attenuation;
            float electron_src_weight;
            float electron_fitted_dmax;
            float jaw_transmission;
            float mlc_transmission;

            bool has_xjaws;
            bool has_yjaws;

            float * kernel;

        } model;

		float collimator_angle;
        float mu;

		float sinca, cosca; // Cached collimator angle trig functions

		MLCPair * mlc;  // MLC leaf pairs
        int n_mlc_pairs;    // Number of leaf pairs

        __host__ IMRTBeam(IMRTBeam * h_beam);
		__host__ IMRTBeam(float * iso, float gantry_angle, float couch_angle, float collimator_angle, const Model * model);

        __device__ float headTransmission(const PointXYZ* point_xyz, const float iso_to_source, const float source_sigma);

		__device__ void offAxisFactors(const PointXYZ * point_xyz, float * off_axis_factor, float * off_axis_softening);

        /** Tilt an IMAGE-space tangent vector such that its z points toward the source */
        __device__ void kernelTilt(const PointXYZ * vox_img_xyz, PointXYZ * vec_img);

		/** Takes collimator angle into account */
		__device__ void pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head);

		/** Takes collimator angle into account */
        __device__ void pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img);

};

class IMRTDose : public CudaDose{

    public:

		using CudaDose::CudaDose;

        __host__ IMRTDose(CudaDose * h_dose);

};

__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray, float * ElectronArray);
__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, Texture3D TERMATexture, Texture3D DensityTexture, float * ElectronArray);

void photon_dose_cuda(int gpu_id, IMRTDose * h_dose, IMRTBeam * h_beam);

#endif