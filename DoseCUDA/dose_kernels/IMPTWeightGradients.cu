#include "IMPTWeightGradients.cuh"
#include "MemoryClasses.h"


/**
 * Reverse-mode product for spot monitor units only.
 *
 * This intentionally mirrors pencilBeamKernel's per-voxel physics while
 * leaving the original forward implementation untouched.  Each voxel/layer
 * thread contributes its scalar dose adjoint to every spot in that layer.
 */
__global__ void pencilBeamWeightVJPKernel(
    IMPTDose *dose,
    IMPTBeam *beam,
    const float *dose_adjoint,
    float *weight_gradient)
{
    PointIJK vox_ijk;
    vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
    vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
    vox_ijk.i = (threadIdx.z + (blockIdx.z * blockDim.z)) / beam->n_layers;
    const unsigned layer_id =
        (threadIdx.z + (blockIdx.z * blockDim.z)) % beam->n_layers;

    if (layer_id >= beam->n_layers || !dose->pointIJKWithinImage(&vox_ijk)) {
        return;
    }

    const unsigned vox_index = dose->pointIJKtoIndex(&vox_ijk);
    const float voxel_adjoint = dose_adjoint[vox_index];
    if (voxel_adjoint == 0.0f) {
        return;
    }

    const float wet = dose->WETArray[vox_index] * 10.0f;
    if (wet > (1.1f * beam->layers[layer_id].r80)) {
        return;
    }

    PointXYZ vox_xyz, vox_head_xyz;
    dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

    float sigma_ms, idd;
    beam->interpolateProtonLUT(wet, &idd, &sigma_ms, layer_id);

    const float distance_to_source = beam->pointXYZDistanceToSource(&vox_xyz);
    const float sigma_total = beam->sigmaAir(wet, distance_to_source, layer_id) + sigma_ms;

    float sigma_halo, halo_weight;
    beam->nuclearHalo(wet, &sigma_halo, &halo_weight, layer_id);
    const float sigma_halo_total = hypotf(sigma_total, sigma_halo);

    const float primary_dose_factor =
        (1.0f - halo_weight) * idd / (2.0f * CUDART_PI_F * sqr(sigma_total));
    const float halo_dose_factor =
        halo_weight * idd / (2.0f * CUDART_PI_F * sqr(sigma_halo_total));
    const float primary_scal = sigma_total ? -0.5f / sqr(sigma_total) : -INFINITY;
    const float halo_scal =
        sigma_halo_total ? -0.5f / sqr(sigma_halo_total) : -INFINITY;

    beam->pointXYZImageToHead(&vox_xyz, &vox_head_xyz);

    const Layer &layer = beam->layers[layer_id];
    const int spot_end = layer.spot_start + layer.n_spots;
    for (int spot_id = layer.spot_start; spot_id < spot_end; ++spot_id) {
        const Spot &spot = beam->spots[spot_id];
        const float distance_to_cax_sqr = beam->caxDistance(spot, vox_head_xyz);
        const float primary_dose =
            primary_dose_factor * expf(primary_scal * distance_to_cax_sqr);
        const float halo_dose =
            halo_dose_factor * expf(halo_scal * distance_to_cax_sqr);

        atomicAdd(
            &weight_gradient[spot_id],
            voxel_adjoint * (primary_dose + halo_dose));
    }
}


void proton_spot_weight_vjp_cuda(
    int gpu_id,
    IMPTDose *h_dose,
    IMPTBeam *h_beam,
    const float *h_dose_adjoint,
    float *h_weight_gradient)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));

    IMPTDose d_dose(h_dose);
    IMPTBeam d_beam(h_beam);

    DevicePointer<float> WETArray(h_dose->WETArray, h_dose->num_voxels);
    DevicePointer<float> DoseAdjoint(h_dose_adjoint, h_dose->num_voxels);
    DevicePointer<float> WeightGradient(MemoryTag::Zeroed(), h_beam->n_spots);

    d_dose.WETArray = WETArray.get();

    DevicePointer<Layer> LayerArray(h_beam->layers, h_beam->n_layers);
    DevicePointer<Spot> SpotArray(h_beam->spots, h_beam->n_spots);
    DevicePointer<float> DivergenceParams(
        h_beam->divergence_params,
        h_beam->dvp_len * h_beam->n_energies);
    DevicePointer<float> LUTDepths(
        h_beam->lut_depths,
        h_beam->lut_len * h_beam->n_energies);
    DevicePointer<float> LUTSigmas(
        h_beam->lut_sigmas,
        h_beam->lut_len * h_beam->n_energies);
    DevicePointer<float> LUTIDDs(
        h_beam->lut_idds,
        h_beam->lut_len * h_beam->n_energies);

    d_beam.layers = LayerArray.get();
    d_beam.spots = SpotArray.get();
    d_beam.divergence_params = DivergenceParams.get();
    d_beam.lut_depths = LUTDepths.get();
    d_beam.lut_sigmas = LUTSigmas.get();
    d_beam.lut_idds = LUTIDDs.get();

    DevicePointer<IMPTBeam> d_beam_ptr(&d_beam);
    DevicePointer<IMPTDose> d_dose_ptr(&d_dose);

    const dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    const dim3 dimGrid(
        (d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH,
        (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH,
        ((d_dose.img_sz.i * static_cast<int>(d_beam.n_layers))
            + TILE_WIDTH - 1) / TILE_WIDTH);

    pencilBeamWeightVJPKernel<<<dimGrid, dimBlock>>>(
        d_dose_ptr,
        d_beam_ptr,
        DoseAdjoint.get(),
        WeightGradient.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        h_weight_gradient,
        WeightGradient.get(),
        h_beam->n_spots * sizeof(float),
        cudaMemcpyDeviceToHost));
}
