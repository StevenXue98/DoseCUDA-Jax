#include "./IMPTClasses.cuh"
#include "MemoryClasses.h"


__host__ IMPTBeam::IMPTBeam(IMPTBeam * h_beam) : CudaBeam(h_beam) {
	this->model = h_beam->model;
	this->n_energies = h_beam->n_energies;
	this->n_layers = h_beam->n_layers;
	this->n_spots = h_beam->n_spots;
	this->dvp_len = h_beam->dvp_len;
	this->lut_len = h_beam->lut_len;
}

__host__ IMPTBeam::IMPTBeam(float * iso, float gantry_angle, float couch_angle, const Model * model) : CudaBeam(iso, gantry_angle, couch_angle, model->sourceDistance()) {
	this->model = *model;
}

/** @brief Count spots in a subarray
 * 	@param spots
 * 		Spots array
 * 	@param n_spots
 * 		Size of the spots array
 * 	@param start
 * 		Beginning index
 * 	@param energy
 * 		Energy to count
 * 	@returns The number of spots starting from @p spots with energy ID @p energy
 */
static int count_spots(const Spot spots[], int n_spots, int start, int energy) {

	int end;

	for (end = start; end < n_spots && spots[end].energy_id == energy; end++);
	return end - start;
}

__host__ void IMPTBeam::importLayers(){

	int spot_start = 0, spot_count;

	this->n_layers = 0;
	for (int energy_id = 0; energy_id < this->n_energies; energy_id++) {
		spot_count = count_spots(this->spots, this->n_spots, spot_start, energy_id);
		if (!spot_count) {
			/* No spots in this layer */
			continue;
		}

		Layer &layer = this->layers[this->n_layers];

		layer.spot_start = spot_start;
		layer.n_spots = spot_count;
		layer.energy_id = energy_id;

		layer.r80 = this->divergence_params[this->dvp_len * energy_id + 1];
		layer.energy = this->divergence_params[this->dvp_len * energy_id];

		// printf("Layer %d: %d spots, r80 = %.2f\n", this->n_layers, layer.n_spots, layer.r80);

		this->n_layers++;

		spot_start += spot_count;
	}

	// printf("%d layers in beam.\n", this->n_layers);

}

__device__ void IMPTBeam::interpolateProtonLUT(float wet, float * idd, float * sigma, unsigned layer_id){

	const Layer &layer = this->layers[layer_id];
	const float *depths, *sigmas, *idds;

	depths = this->lut_depths + layer.energy_id * this->lut_len;
	sigmas = this->lut_sigmas + layer.energy_id * this->lut_len;
	idds = this->lut_idds + layer.energy_id * this->lut_len;

	int i = lowerBound(depths, this->lut_len, wet);

	if (i == this->lut_len) {
		*idd = idds[i - 1];
		*sigma = sigmas[i - 1];
	} else if (i == 0) {
		*idd = idds[i];
		*sigma = sigmas[i];
	} else {
		auto factor = (wet - depths[i - 1]) / (depths[i] - depths[i - 1]);
		*idd = fmaf(idds[i] - idds[i - 1], factor, idds[i - 1]);
		*sigma = fmaf(sigmas[i] - sigmas[i - 1], factor, sigmas[i - 1]);
	}

}

__device__ float IMPTBeam::sigmaAir(float wet, float distance_to_source, unsigned layer_id) {

	Layer &layer = this->layers[layer_id];
	float d = distance_to_source - wet + (0.7f * layer.r80);
	const float *coef = &this->divergence_params[this->dvp_len * layer.energy_id + 2];

	return fmaf(fmaf(coef[0], d, coef[1]), d, coef[2]);

}

__device__ void IMPTBeam::nuclearHalo(float wet, float * halo_sigma, float * halo_weight, unsigned layer_id) {

	Layer &layer = this->layers[layer_id];

	wet = clamp(wet, 0.1f, layer.r80 - 0.1f);

	float halo_sigma_ = 2.85f + (0.0014f * layer.r80 * logf(wet + 3.0f)) +
	(0.06f * wet) - (7.4e-5f * sqr(wet)) -
	((0.22f * layer.r80) / sqr(wet - layer.r80 - 5.0f));

	float halo_weight_ = 0.052f * logf(1.13f + (wet / (11.2f - (0.023f * layer.r80)))) +
	(0.35f * ((0.0017f * sqr(layer.r80)) - layer.r80) / (sqr(layer.r80 + 3.0f) - sqr(wet))) -
	(1.61e-9f * wet * sqr(layer.r80 + 3.0f));

	*halo_sigma = fmaxf(halo_sigma_, 0.0f);
	*halo_weight = clamp(halo_weight_, 0.0f, 0.9f);

}

__device__ float IMPTBeam::caxDistance(const Spot &spot, const PointXYZ &vox)
{
	PointXYZ tangent = { spot.x / model.vsadx, spot.y / model.vsady, -1.0f };
	PointXYZ ray = { vox.x - spot.x, vox.y - spot.y, vox.z };

	const auto tansqr = xyz_dotproduct(tangent, tangent);
	const auto raysqr = xyz_dotproduct(ray, ray);
	const auto dotprd = xyz_dotproduct(tangent, ray);

	return raysqr - dotprd * dotprd / tansqr;
}


__host__ IMPTDose::IMPTDose(CudaDose * h_dose) : CudaDose(h_dose) {}


__global__ void smoothRayKernel(IMPTDose * dose, CudaBeam * beam, float * SmoothedWETArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	int vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	PointXYZ uvec;
	beam->unitVectorToSource(&vox_xyz, &uvec);

	float step_length = 1.0f;

	PointXYZ vox_head_xyz;
	PointXYZ vox_head_xyz_conv;
	PointXYZ vox_image_xyz_conv;
	PointIJK vox_ijk_conv;
	int vox_index_conv;

	float center_wet = dose->WETArray[vox_index];
	float dr = 1.0;
	float conv_wet_sum = dose->WETArray[vox_index];
	int n_voxels = 1;

	beam->pointXYZImageToHead(&vox_xyz, &vox_head_xyz);

	for (int i=0; i<6; i++){
		float sinx, cosx;

		dr = 1.0f;
		sincosf((float)i * CUDART_PI_F / 3.0f, &sinx, &cosx);

		while ((dr < (center_wet * 10.0f)) && (dr < 10.0f)){

			vox_head_xyz_conv.x = vox_head_xyz.x + (dr * cosx);
			vox_head_xyz_conv.y = vox_head_xyz.y + (dr * sinx);
			vox_head_xyz_conv.z = vox_head_xyz.z;

			beam->pointXYZHeadToImage(&vox_head_xyz_conv, &vox_image_xyz_conv);
			dose->pointXYZtoIJK(&vox_image_xyz_conv, &vox_ijk_conv, beam);

			if(dose->pointIJKWithinImage(&vox_ijk_conv)){
				vox_index_conv = dose->pointIJKtoIndex(&vox_ijk_conv);
				conv_wet_sum += dose->WETArray[vox_index_conv];
				n_voxels += 1;
			}

			dr += step_length;
		}
	}

	SmoothedWETArray[vox_index] = conv_wet_sum / (float)n_voxels;

	__syncthreads();

}

__global__ void pencilBeamKernel(IMPTDose * dose, IMPTBeam * beam){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = (threadIdx.z + (blockIdx.z * blockDim.z)) / beam->n_layers;
	unsigned layer_id = (threadIdx.z + (blockIdx.z * blockDim.z)) % beam->n_layers;

	if (layer_id >= beam->n_layers){
		return;
	}

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	unsigned vox_index = dose->pointIJKtoIndex(&vox_ijk);

	float wet = dose->WETArray[vox_index] * 10.0f;

	if (wet > (1.1f * beam->layers[layer_id].r80)){
		return;
	}

	PointXYZ vox_xyz, vox_head_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	float sigma_ms, idd;
	beam->interpolateProtonLUT(wet, &idd, &sigma_ms, layer_id);

	float distance_to_source = beam->pointXYZDistanceToSource(&vox_xyz);
	float sigma_total = beam->sigmaAir(wet, distance_to_source, layer_id) + sigma_ms;

	float sigma_halo, halo_weight;
	beam->nuclearHalo(wet, &sigma_halo, &halo_weight, layer_id);
	float sigma_halo_total = hypotf(sigma_total, sigma_halo);

	Layer &layer = beam->layers[layer_id];

	float primary_dose_factor = (1.0f - halo_weight) * idd / (2.0f * CUDART_PI_F * sqr(sigma_total));
	float halo_dose_factor = halo_weight * idd / (2.0f * CUDART_PI_F * sqr(sigma_halo_total));

	float total_dose = 0.0f, primary_dose, halo_dose;

	beam->pointXYZImageToHead(&vox_xyz, &vox_head_xyz);

	const int spot_end = layer.spot_start + layer.n_spots;

	const float primary_scal = sigma_total ? -0.5f / sqr(sigma_total) : -INFINITY;
	const float halo_scal = sigma_halo_total ? -0.5f / sqr(sigma_halo_total) : -INFINITY;

	for (int spot_id=layer.spot_start; spot_id < spot_end; spot_id++){

		const Spot &spot = beam->spots[spot_id];
		auto distance_to_cax_sqr = beam->caxDistance(spot, vox_head_xyz);

		primary_dose = primary_dose_factor * expf(primary_scal * distance_to_cax_sqr);
		halo_dose = halo_dose_factor * expf(halo_scal * distance_to_cax_sqr);

		total_dose = fmaf(spot.mu, primary_dose + halo_dose, total_dose);

	}

	atomicAdd(&dose->DoseArray[vox_index], total_dose);

	__syncthreads();

}


void proton_raytrace_cuda(int gpu_id, CudaDose * h_dose, CudaBeam  * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMPTDose d_dose(h_dose);
	CudaBeam d_beam(h_beam);

	DevicePointer<float> WETArray(h_dose->num_voxels);
	DevicePointer<float> SmoothedWETArray(d_dose.num_voxels);

	d_dose.WETArray = WETArray.get();

	DevicePointer<IMPTDose> d_dose_ptr(&d_dose);
	DevicePointer<CudaBeam> d_beam_ptr(&d_beam);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear);

	rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
	smoothRayKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, SmoothedWETArray);

	CUDA_CHECK(cudaMemcpy(h_dose->WETArray, SmoothedWETArray.get(), d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}

void proton_spot_cuda(int gpu_id, IMPTDose * h_dose, IMPTBeam  * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMPTDose d_dose(h_dose);
	IMPTBeam d_beam(h_beam);

	DevicePointer<float> DoseArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> DensityArray(h_dose->DensityArray, h_dose->num_voxels);
	DevicePointer<float> WETArray(h_dose->WETArray, h_dose->num_voxels);

	d_dose.DoseArray = DoseArray.get();
	d_dose.DensityArray = DensityArray.get();
	d_dose.WETArray = WETArray.get();

	DevicePointer<Layer> LayerArray(h_beam->layers, h_beam->n_layers);
	DevicePointer<Spot> SpotArray(h_beam->spots, h_beam->n_spots);
	DevicePointer<float> DivergenceParams(h_beam->divergence_params, h_beam->dvp_len * h_beam->n_energies);
	DevicePointer<float> LUTDepths(h_beam->lut_depths, h_beam->lut_len * h_beam->n_energies);
	DevicePointer<float> LUTSigmas(h_beam->lut_sigmas, h_beam->lut_len * h_beam->n_energies);
	DevicePointer<float> LUTIDDs(h_beam->lut_idds, h_beam->lut_len * h_beam->n_energies);

	d_beam.layers = LayerArray.get();
	d_beam.spots = SpotArray.get();
	d_beam.divergence_params = DivergenceParams.get();
	d_beam.lut_depths = LUTDepths.get();
	d_beam.lut_sigmas = LUTSigmas.get();
	d_beam.lut_idds = LUTIDDs.get();

	DevicePointer<IMPTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMPTDose> d_dose_ptr(&d_dose);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, ((d_dose.img_sz.i * (int)d_beam.n_layers) + TILE_WIDTH - 1) / TILE_WIDTH);

	pencilBeamKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
