#include "IMPTInfluenceMatrix.cuh"

#include <cublas_v2.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "CudaClasses.cuh"
#include "MemoryClasses.h"

namespace {

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
    }
}

struct BlasHandle {
    cublasHandle_t value = nullptr;
    void initialize() { blas_check(cublasCreate(&value)); }
    ~BlasHandle() { if (value) cublasDestroy(value); }
};

__global__ void matrix_loss_and_adjoint(
    const double *dose, double *adjoint, double *objective,
    const unsigned char *target, const unsigned char *oar,
    const unsigned char *normal, int n_voxels, double prescription,
    double oar_limit, double normal_limit, double target_factor,
    double oar_factor, double normal_factor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_voxels) return;
    const double value = dose[i];
    double term = 0.0;
    double derivative = 0.0;
    if (target[i]) {
        const double error = value - prescription;
        term += target_factor * error * error;
        derivative += 2.0 * target_factor * error;
    }
    if (oar[i]) {
        const double excess = fmax(value - oar_limit, 0.0);
        term += oar_factor * excess * excess;
        derivative += 2.0 * oar_factor * excess;
    }
    if (normal[i]) {
        const double excess = fmax(value - normal_limit, 0.0);
        term += normal_factor * excess * excess;
        derivative += 2.0 * normal_factor * excess;
    }
    adjoint[i] = derivative;
    if (term != 0.0) atomicAdd(objective, term);
}

}  // namespace

struct GPUInfluenceMatrix::Impl {
    BlasHandle blas;
    std::unique_ptr<DevicePointer<double>> matrix;
    std::unique_ptr<DevicePointer<double>> weights;
    std::unique_ptr<DevicePointer<double>> dose;
    std::unique_ptr<DevicePointer<double>> adjoint;
    std::unique_ptr<DevicePointer<double>> gradient;
    std::unique_ptr<DevicePointer<double>> objective;
    std::unique_ptr<DevicePointer<unsigned char>> target;
    std::unique_ptr<DevicePointer<unsigned char>> oar;
    std::unique_ptr<DevicePointer<unsigned char>> normal;
    double prescription;
    double oar_limit;
    double normal_limit;
    double target_factor;
    double oar_factor;
    double normal_factor;
};

GPUInfluenceMatrix::GPUInfluenceMatrix(
    int gpu_id, const double *matrix, int n_voxels, int n_spots,
    const unsigned char *target, const unsigned char *oar,
    const unsigned char *normal, double prescription, double oar_limit,
    double normal_limit, double oar_weight, double normal_weight)
    : impl_(nullptr), gpu_id_(gpu_id), n_voxels_(n_voxels), n_spots_(n_spots) {
    if (!matrix || !target || !oar || !normal || n_voxels <= 0 ||
        n_spots <= 0 || !std::isfinite(prescription) || prescription <= 0.0 ||
        !std::isfinite(oar_limit) || oar_limit < 0.0 ||
        !std::isfinite(normal_limit) || normal_limit < 0.0 ||
        !std::isfinite(oar_weight) || oar_weight < 0.0 ||
        !std::isfinite(normal_weight) || normal_weight < 0.0) {
        throw std::invalid_argument("invalid GPU influence-matrix settings");
    }
    int target_count = 0, oar_count = 0, normal_count = 0;
    for (int i = 0; i < n_voxels; ++i) {
        if (target[i] + oar[i] + normal[i] > 1) {
            throw std::invalid_argument("structure masks must be disjoint");
        }
        target_count += target[i] != 0;
        oar_count += oar[i] != 0;
        normal_count += normal[i] != 0;
    }
    if (!target_count || !oar_count || !normal_count) {
        throw std::invalid_argument("structure masks must be nonempty");
    }
    CUDA_CHECK(cudaSetDevice(gpu_id));
    std::unique_ptr<Impl> impl(new Impl());
    impl->blas.initialize();
    impl->matrix.reset(new DevicePointer<double>(
        matrix, static_cast<size_t>(n_voxels) * n_spots));
    impl->weights.reset(new DevicePointer<double>(n_spots));
    impl->dose.reset(new DevicePointer<double>(n_voxels));
    impl->adjoint.reset(new DevicePointer<double>(n_voxels));
    impl->gradient.reset(new DevicePointer<double>(n_spots));
    impl->objective.reset(new DevicePointer<double>(1));
    impl->target.reset(new DevicePointer<unsigned char>(target, n_voxels));
    impl->oar.reset(new DevicePointer<unsigned char>(oar, n_voxels));
    impl->normal.reset(new DevicePointer<unsigned char>(normal, n_voxels));
    impl->prescription = prescription;
    impl->oar_limit = oar_limit;
    impl->normal_limit = normal_limit;
    const double dose_factor = 1.0 / (prescription * prescription);
    impl->target_factor = dose_factor / target_count;
    impl->oar_factor = dose_factor * oar_weight / oar_count;
    impl->normal_factor = dose_factor * normal_weight / normal_count;
    impl_ = impl.release();
}

GPUInfluenceMatrix::~GPUInfluenceMatrix() {
    cudaSetDevice(gpu_id_);
    delete impl_;
}

double GPUInfluenceMatrix::value_and_gradient(
    const double *weights, double *gradient, double *dose_host) {
    CUDA_CHECK(cudaSetDevice(gpu_id_));
    CUDA_CHECK(cudaMemcpy(impl_->weights->get(), weights,
                          n_spots_ * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    // NumPy's C-order (voxels, spots) matrix is column-major (spots, voxels)
    // to cuBLAS. Transpose for dose, then use it directly for the adjoint.
    blas_check(cublasDgemv(impl_->blas.value, CUBLAS_OP_T,
                           n_spots_, n_voxels_, &one, impl_->matrix->get(),
                           n_spots_, impl_->weights->get(), 1, &zero,
                           impl_->dose->get(), 1));
    CUDA_CHECK(cudaMemset(impl_->objective->get(), 0, sizeof(double)));
    matrix_loss_and_adjoint<<<(n_voxels_ + 255) / 256, 256>>>(
        impl_->dose->get(), impl_->adjoint->get(), impl_->objective->get(),
        impl_->target->get(), impl_->oar->get(), impl_->normal->get(),
        n_voxels_, impl_->prescription, impl_->oar_limit, impl_->normal_limit,
        impl_->target_factor, impl_->oar_factor, impl_->normal_factor);
    CUDA_CHECK(cudaGetLastError());
    blas_check(cublasDgemv(impl_->blas.value, CUBLAS_OP_N,
                           n_spots_, n_voxels_, &one, impl_->matrix->get(),
                           n_spots_, impl_->adjoint->get(), 1, &zero,
                           impl_->gradient->get(), 1));
    double objective;
    CUDA_CHECK(cudaMemcpy(&objective, impl_->objective->get(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradient, impl_->gradient->get(),
                          n_spots_ * sizeof(double), cudaMemcpyDeviceToHost));
    if (dose_host) {
        CUDA_CHECK(cudaMemcpy(dose_host, impl_->dose->get(),
                              n_voxels_ * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (!std::isfinite(objective)) {
        throw std::runtime_error("nonfinite GPU influence-matrix objective");
    }
    return objective;
}

void GPUInfluenceMatrix::dose_only(const double *weights, double *dose_host) {
    CUDA_CHECK(cudaSetDevice(gpu_id_));
    CUDA_CHECK(cudaMemcpy(impl_->weights->get(), weights,
                          n_spots_ * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    blas_check(cublasDgemv(impl_->blas.value, CUBLAS_OP_T,
                           n_spots_, n_voxels_, &one, impl_->matrix->get(),
                           n_spots_, impl_->weights->get(), 1, &zero,
                           impl_->dose->get(), 1));
    CUDA_CHECK(cudaMemcpy(dose_host, impl_->dose->get(),
                          static_cast<size_t>(n_voxels_) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPUInfluenceMatrix::weight_vjp(
    const double *dose_adjoint, double *weight_gradient) {
    CUDA_CHECK(cudaSetDevice(gpu_id_));
    CUDA_CHECK(cudaMemcpy(impl_->adjoint->get(), dose_adjoint,
                          static_cast<size_t>(n_voxels_) * sizeof(double),
                          cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    blas_check(cublasDgemv(impl_->blas.value, CUBLAS_OP_N,
                           n_spots_, n_voxels_, &one, impl_->matrix->get(),
                           n_spots_, impl_->adjoint->get(), 1, &zero,
                           impl_->gradient->get(), 1));
    CUDA_CHECK(cudaMemcpy(weight_gradient, impl_->gradient->get(),
                          n_spots_ * sizeof(double), cudaMemcpyDeviceToHost));
}
