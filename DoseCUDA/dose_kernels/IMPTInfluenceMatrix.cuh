#ifndef IMPT_INFLUENCE_MATRIX_H
#define IMPT_INFLUENCE_MATRIX_H

#include <cstddef>

class GPUInfluenceMatrix {
public:
    GPUInfluenceMatrix(int gpu_id, const double *matrix, int n_voxels,
                       int n_spots, const unsigned char *target,
                       const unsigned char *oar, const unsigned char *normal,
                       double prescription, double oar_limit,
                       double normal_limit, double oar_weight,
                       double normal_weight);
    ~GPUInfluenceMatrix();

    GPUInfluenceMatrix(const GPUInfluenceMatrix &) = delete;
    GPUInfluenceMatrix &operator=(const GPUInfluenceMatrix &) = delete;

    int spot_count() const { return n_spots_; }
    int voxel_count() const { return n_voxels_; }
    double value_and_gradient(const double *weights, double *gradient,
                              double *dose_host = nullptr);
    void dose_only(const double *weights, double *dose_host);
    void weight_vjp(const double *dose_adjoint, double *weight_gradient);

private:
    struct Impl;
    Impl *impl_;
    int gpu_id_;
    int n_voxels_;
    int n_spots_;
};

#endif
