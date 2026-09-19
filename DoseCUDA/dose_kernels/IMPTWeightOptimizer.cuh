#ifndef IMPT_WEIGHT_OPTIMIZER_H
#define IMPT_WEIGHT_OPTIMIZER_H

#include <vector>
#include "IMPTClasses.cuh"

struct IMPTWeightOptimizerResult {
    std::vector<float> weights;
    float objective;
    float projected_gradient;
    int iterations;
    int forward_evaluations;
    int gradient_evaluations;
    bool converged;
};

// Experimental float32, matrix-free, device-resident spot-weight optimization.
// The caller owns host beam data and supplies weights in each beam's sorted order.
IMPTWeightOptimizerResult optimize_impt_weights_cuda(
    int gpu_id,
    const std::vector<IMPTDose *> &doses,
    const std::vector<IMPTBeam *> &beams,
    const unsigned char *target_mask,
    const unsigned char *oar_mask,
    const unsigned char *normal_mask,
    float prescription,
    float oar_limit,
    float normal_limit,
    float dose_scale,
    const float *initial_weights,
    int max_iterations,
    float gradient_tolerance,
    const char *method);

#endif
