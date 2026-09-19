#ifndef IMPT_WEIGHT_GRADIENTS_H
#define IMPT_WEIGHT_GRADIENTS_H

#include "IMPTClasses.cuh"

__global__ void pencilBeamWeightVJPKernel(
    IMPTDose *dose,
    IMPTBeam *beam,
    const float *dose_adjoint,
    float *weight_gradient);

/** Apply the transpose of the fixed-geometry proton spot-dose operator.
 *
 * Given a voxel adjoint dL/dDose, this computes one derivative dL/dMU for
 * every spot.  Geometry, WET, spot positions, and energy layers are treated as
 * constants, matching the forward proton_spot_cuda calculation.
 */
void proton_spot_weight_vjp_cuda(
    int gpu_id,
    IMPTDose *h_dose,
    IMPTBeam *h_beam,
    const float *h_dose_adjoint,
    float *h_weight_gradient);

#endif
