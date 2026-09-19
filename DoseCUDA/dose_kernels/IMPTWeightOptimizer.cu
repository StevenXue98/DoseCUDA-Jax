#include "IMPTWeightOptimizer.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "IMPTWeightGradients.cuh"
#include "MemoryClasses.h"

namespace {

struct BeamBuffers {
    IMPTBeam beam;
    DevicePointer<Layer> layers;
    DevicePointer<Spot> spots;
    DevicePointer<float> divergence;
    DevicePointer<float> depths;
    DevicePointer<float> sigmas;
    DevicePointer<float> idds;
    DevicePointer<float> wet;
    DevicePointer<IMPTBeam> beam_ptr;
    IMPTDose dose;
    DevicePointer<IMPTDose> dose_ptr;
    int offset;

    BeamBuffers(IMPTBeam *host_beam, IMPTDose *host_dose, int weight_offset,
                float *device_dose)
        : beam(host_beam),
          layers(host_beam->layers, host_beam->n_layers),
          spots(host_beam->spots, host_beam->n_spots),
          divergence(host_beam->divergence_params,
                     host_beam->dvp_len * host_beam->n_energies),
          depths(host_beam->lut_depths,
                 host_beam->lut_len * host_beam->n_energies),
          sigmas(host_beam->lut_sigmas,
                 host_beam->lut_len * host_beam->n_energies),
          idds(host_beam->lut_idds,
               host_beam->lut_len * host_beam->n_energies),
          wet(host_dose->WETArray, host_dose->num_voxels),
          beam_ptr(1),
          dose(host_dose),
          dose_ptr(1),
          offset(weight_offset) {
        beam.layers = layers.get();
        beam.spots = spots.get();
        beam.divergence_params = divergence.get();
        beam.lut_depths = depths.get();
        beam.lut_sigmas = sigmas.get();
        beam.lut_idds = idds.get();
        dose.DoseArray = device_dose;
        dose.WETArray = wet.get();
        CUDA_CHECK(cudaMemcpy(beam_ptr.get(), &beam, sizeof(IMPTBeam),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dose_ptr.get(), &dose, sizeof(IMPTDose),
                              cudaMemcpyHostToDevice));
    }
};

__global__ void set_spot_weights(Spot *spots, const float *weights,
                                 int offset, int n_spots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_spots) spots[i].mu = weights[offset + i];
}

__global__ void loss_and_adjoint(const float *raw_dose, float *adjoint,
                                 float *objective, const unsigned char *target,
                                 const unsigned char *oar,
                                 const unsigned char *normal, int n_voxels,
                                 float dose_scale, float prescription,
                                 float oar_limit, float normal_limit,
                                 float target_factor, float oar_factor,
                                 float normal_factor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_voxels) return;
    const float value = raw_dose[i] * dose_scale;
    float term = 0.0f;
    float derivative = 0.0f;
    if (target[i]) {
        const float error = value - prescription;
        term += target_factor * error * error;
        derivative += 2.0f * target_factor * error;
    }
    if (oar[i]) {
        const float excess = fmaxf(value - oar_limit, 0.0f);
        term += oar_factor * excess * excess;
        derivative += 2.0f * oar_factor * excess;
    }
    if (normal[i]) {
        const float excess = fmaxf(value - normal_limit, 0.0f);
        term += normal_factor * excess * excess;
        derivative += 2.0f * normal_factor * excess;
    }
    adjoint[i] = derivative * dose_scale;
    if (term != 0.0f) atomicAdd(objective, term);
}

__global__ void projected_step(const float *point, const float *gradient,
                               float *candidate, float step, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) candidate[i] = fmaxf(point[i] - step * gradient[i], 0.0f);
}

__global__ void direction_step(const float *point, const float *direction,
                               float *candidate, float step, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) candidate[i] = fmaxf(point[i] + step * direction[i], 0.0f);
}

__global__ void cg_beta_statistics(const float *point, const float *gradient,
                                   const float *previous_gradient,
                                   float *statistics, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float current = point[i] > 0.0f || gradient[i] < 0.0f
        ? gradient[i] : 0.0f;
    const float previous = previous_gradient[i];
    atomicAdd(&statistics[0], current * (current - previous));
    atomicAdd(&statistics[1], previous * previous);
}

__global__ void cg_direction(const float *point, const float *gradient,
                             float *previous_gradient, float *direction,
                             float beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float projected = point[i] > 0.0f || gradient[i] < 0.0f
        ? gradient[i] : 0.0f;
    float next = -projected + beta * direction[i];
    if (point[i] <= 0.0f && next < 0.0f) next = 0.0f;
    direction[i] = next;
    previous_gradient[i] = projected;
}

__global__ void direction_slope(const float *gradient, const float *direction,
                                float *slope, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(slope, gradient[i] * direction[i]);
}

__global__ void step_model(const float *point, const float *gradient,
                           const float *candidate, float *model, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float difference = candidate[i] - point[i];
    atomicAdd(&model[0], gradient[i] * difference);
    atomicAdd(&model[1], difference * difference);
}

__global__ void extrapolate(const float *point, const float *previous,
                            float *output, float momentum, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = fmaxf(point[i] + momentum * (point[i] - previous[i]), 0.0f);
    }
}

__global__ void projected_norm(const float *point, const float *gradient,
                               int *maximum, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float stationarity = fabsf(
        point[i] - fmaxf(point[i] - gradient[i], 0.0f));
    atomicMax(maximum, __float_as_int(stationarity));
}

float device_scalar(const float *pointer) {
    float value;
    CUDA_CHECK(cudaMemcpy(&value, pointer, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

}  // namespace

IMPTWeightOptimizerResult optimize_impt_weights_cuda(
    int gpu_id, const std::vector<IMPTDose *> &host_doses,
    const std::vector<IMPTBeam *> &host_beams,
    const unsigned char *target_mask, const unsigned char *oar_mask,
    const unsigned char *normal_mask, float prescription, float oar_limit,
    float normal_limit, float dose_scale, const float *initial_weights,
    int max_iterations, float gradient_tolerance, const char *method) {
    if (host_beams.empty() || host_beams.size() != host_doses.size() ||
        max_iterations <= 0 ||
        !(gradient_tolerance > 0.0f) || !(prescription > 0.0f) ||
        !(dose_scale > 0.0f) || !std::isfinite(oar_limit) ||
        !std::isfinite(normal_limit) || oar_limit < 0.0f ||
        normal_limit < 0.0f ||
        (std::strcmp(method, "pg") && std::strcmp(method, "fista")
         && std::strcmp(method, "cg") && std::strcmp(method, "lbfgs"))) {
        throw std::invalid_argument("invalid CUDA weight optimizer settings");
    }
    CUDA_CHECK(cudaSetDevice(gpu_id));
    const int n_voxels = static_cast<int>(host_doses[0]->num_voxels);
    int n_spots = 0;
    for (const IMPTBeam *beam : host_beams) n_spots += beam->n_spots;
    if (n_spots <= 0) throw std::invalid_argument("no spots to optimize");

    int target_count = 0, oar_count = 0, normal_count = 0;
    for (int i = 0; i < n_voxels; ++i) {
        target_count += target_mask[i] != 0;
        oar_count += oar_mask[i] != 0;
        normal_count += normal_mask[i] != 0;
    }
    if (!target_count || !oar_count || !normal_count) {
        throw std::invalid_argument("target, OAR, and normal masks must be nonempty");
    }

    DevicePointer<float> raw_dose(MemoryTag::Zeroed(), n_voxels);
    DevicePointer<float> adjoint(n_voxels);
    DevicePointer<float> objective(MemoryTag::Zeroed(), 1);
    DevicePointer<unsigned char> target(target_mask, n_voxels);
    DevicePointer<unsigned char> oar(oar_mask, n_voxels);
    DevicePointer<unsigned char> normal(normal_mask, n_voxels);
    DevicePointer<float> x(initial_weights, n_spots);
    DevicePointer<float> old(initial_weights, n_spots);
    DevicePointer<float> point(n_spots);
    DevicePointer<float> candidate(n_spots);
    DevicePointer<float> gradient(MemoryTag::Zeroed(), n_spots);
    DevicePointer<float> previous_gradient(MemoryTag::Zeroed(), n_spots);
    DevicePointer<float> direction(MemoryTag::Zeroed(), n_spots);
    DevicePointer<float> model(MemoryTag::Zeroed(), 2);
    DevicePointer<int> norm(MemoryTag::Zeroed(), 1);
    std::vector<std::unique_ptr<BeamBuffers>> beams;
    int offset = 0;
    for (size_t index = 0; index < host_beams.size(); ++index) {
        IMPTBeam *beam = host_beams[index];
        if (host_doses[index]->num_voxels != static_cast<unsigned>(n_voxels)) {
            throw std::invalid_argument("beam WET volumes must have matching shapes");
        }
        beams.emplace_back(new BeamBuffers(
            beam, host_doses[index], offset, raw_dose.get()));
        offset += beam->n_spots;
    }
    const int weight_blocks = (n_spots + 255) / 256;
    const int voxel_blocks = (n_voxels + 255) / 256;
    const float dose_factor = 1.0f / (prescription * prescription);
    const float target_factor = dose_factor / target_count;
    const float oar_factor = dose_factor / oar_count;
    const float normal_factor = dose_factor / normal_count;
    const dim3 dose_block(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

    int forward_evaluations = 0;
    int gradient_evaluations = 0;
    auto evaluate = [&](const float *weights, bool with_gradient) {
        CUDA_CHECK(cudaMemset(raw_dose.get(), 0, n_voxels * sizeof(float)));
        for (const auto &beam : beams) {
            set_spot_weights<<<(beam->beam.n_spots + 255) / 256, 256>>>(
                beam->spots.get(), weights, beam->offset, beam->beam.n_spots);
            const dim3 dose_grid(
                (beam->dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH,
                (beam->dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH,
                (beam->dose.img_sz.i * beam->beam.n_layers + TILE_WIDTH - 1)
                    / TILE_WIDTH);
            pencilBeamKernel<<<dose_grid, dose_block>>>(
                beam->dose_ptr.get(), beam->beam_ptr.get());
        }
        CUDA_CHECK(cudaMemset(objective.get(), 0, sizeof(float)));
        loss_and_adjoint<<<voxel_blocks, 256>>>(
            raw_dose.get(), adjoint.get(), objective.get(), target.get(),
            oar.get(), normal.get(), n_voxels, dose_scale, prescription,
            oar_limit, normal_limit, target_factor, oar_factor, normal_factor);
        ++forward_evaluations;
        if (with_gradient) {
            CUDA_CHECK(cudaMemset(gradient.get(), 0, n_spots * sizeof(float)));
            for (const auto &beam : beams) {
                const dim3 dose_grid(
                    (beam->dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH,
                    (beam->dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH,
                    (beam->dose.img_sz.i * beam->beam.n_layers + TILE_WIDTH - 1)
                        / TILE_WIDTH);
                pencilBeamWeightVJPKernel<<<dose_grid, dose_block>>>(
                    beam->dose_ptr.get(), beam->beam_ptr.get(), adjoint.get(),
                    gradient.get() + beam->offset);
            }
            ++gradient_evaluations;
        }
        CUDA_CHECK(cudaGetLastError());
        const float value = device_scalar(objective.get());
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite CUDA loss");
        return value;
    };

    if (std::strcmp(method, "lbfgs") == 0) {
        const int history_limit = 8;
        std::vector<std::vector<float>> steps, changes;
        std::vector<float> x_host(initial_weights, initial_weights + n_spots);
        std::vector<float> gradient_host(n_spots), previous_gradient_host(n_spots);
        std::vector<float> direction_host(n_spots), candidate_host(n_spots);
        std::vector<float> scratch(n_spots);
        auto dot = [&](const std::vector<float> &a,
                       const std::vector<float> &b) {
            double sum = 0.0;
            for (int i = 0; i < n_spots; ++i) sum += double(a[i]) * b[i];
            return sum;
        };
        float value = evaluate(x.get(), true);
        int iterations = 0;
        float stationarity = 0.0f;
        for (; iterations < max_iterations; ++iterations) {
            CUDA_CHECK(cudaMemcpy(gradient_host.data(), gradient.get(),
                                  n_spots * sizeof(float), cudaMemcpyDeviceToHost));
            stationarity = 0.0f;
            for (int i = 0; i < n_spots; ++i) {
                const float projected = (x_host[i] > 0.0f || gradient_host[i] < 0.0f)
                    ? gradient_host[i] : 0.0f;
                stationarity = std::max(stationarity, fabsf(
                    x_host[i] - fmaxf(x_host[i] - gradient_host[i], 0.0f)));
                scratch[i] = projected;
            }
            if (stationarity <= gradient_tolerance) break;
            std::vector<double> alpha(steps.size()), rho(steps.size());
            for (int j = static_cast<int>(steps.size()) - 1; j >= 0; --j) {
                rho[j] = 1.0 / dot(steps[j], changes[j]);
                alpha[j] = rho[j] * dot(steps[j], scratch);
                for (int i = 0; i < n_spots; ++i) {
                    scratch[i] -= static_cast<float>(alpha[j] * changes[j][i]);
                }
            }
            if (!steps.empty()) {
                const double gamma = dot(steps.back(), changes.back()) /
                    dot(changes.back(), changes.back());
                for (float &component : scratch) component *= static_cast<float>(gamma);
            }
            for (size_t j = 0; j < steps.size(); ++j) {
                const double beta = rho[j] * dot(changes[j], scratch);
                for (int i = 0; i < n_spots; ++i) {
                    scratch[i] += static_cast<float>((alpha[j] - beta) * steps[j][i]);
                }
            }
            for (int i = 0; i < n_spots; ++i) {
                direction_host[i] = -scratch[i];
                if (x_host[i] <= 0.0f && direction_host[i] < 0.0f) {
                    direction_host[i] = 0.0f;
                }
            }
            double descent = dot(gradient_host, direction_host);
            if (descent >= -1.0e-14) {
                for (int i = 0; i < n_spots; ++i) {
                    direction_host[i] = -(x_host[i] > 0.0f || gradient_host[i] < 0.0f
                                          ? gradient_host[i] : 0.0f);
                }
                descent = dot(gradient_host, direction_host);
                steps.clear();
                changes.clear();
            }
            if (descent >= 0.0) break;
            float trial_step = 1.0f;
            float candidate_value = value;
            bool accepted = false;
            for (int attempt = 0; attempt < 28; ++attempt) {
                double slope = 0.0;
                for (int i = 0; i < n_spots; ++i) {
                    candidate_host[i] = fmaxf(
                        x_host[i] + trial_step * direction_host[i], 0.0f);
                    slope += double(gradient_host[i]) *
                        (candidate_host[i] - x_host[i]);
                }
                if (slope >= 0.0) {
                    trial_step *= 0.5f;
                    continue;
                }
                CUDA_CHECK(cudaMemcpy(candidate.get(), candidate_host.data(),
                                      n_spots * sizeof(float), cudaMemcpyHostToDevice));
                candidate_value = evaluate(candidate.get(), false);
                if (candidate_value <= value + 1.0e-4 * slope + 1.0e-9) {
                    accepted = true;
                    break;
                }
                trial_step *= 0.5f;
            }
            if (!accepted) break;
            CUDA_CHECK(cudaMemcpy(x.get(), candidate.get(),
                                  n_spots * sizeof(float), cudaMemcpyDeviceToDevice));
            candidate_value = evaluate(x.get(), true);
            CUDA_CHECK(cudaMemcpy(previous_gradient_host.data(), gradient.get(),
                                  n_spots * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> s(n_spots), y(n_spots);
            for (int i = 0; i < n_spots; ++i) {
                s[i] = candidate_host[i] - x_host[i];
                y[i] = previous_gradient_host[i] - gradient_host[i];
            }
            if (dot(s, y) > 1.0e-10 * std::sqrt(dot(s, s) * dot(y, y))) {
                if (steps.size() == history_limit) {
                    steps.erase(steps.begin());
                    changes.erase(changes.begin());
                }
                steps.push_back(std::move(s));
                changes.push_back(std::move(y));
            }
            x_host.swap(candidate_host);
            value = candidate_value;
        }
        CUDA_CHECK(cudaMemcpy(gradient_host.data(), gradient.get(),
                              n_spots * sizeof(float), cudaMemcpyDeviceToHost));
        stationarity = 0.0f;
        for (int i = 0; i < n_spots; ++i) {
            const float component = fabsf(
                x_host[i] - fmaxf(x_host[i] - gradient_host[i], 0.0f));
            stationarity = std::max(stationarity, component);
        }
        return {x_host, value, stationarity, iterations,
                forward_evaluations, gradient_evaluations,
                stationarity <= gradient_tolerance};
    }

    float best_value = evaluate(x.get(), false);
    float step = 1.0f;
    float t = 1.0f;
    int iterations = 0;
    bool converged = false;
    for (; iterations < max_iterations; ++iterations) {
        const bool accelerated = std::strcmp(method, "fista") == 0;
        const bool conjugate = std::strcmp(method, "cg") == 0;
        const float next_t = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float momentum = accelerated ? (t - 1.0f) / next_t : 0.0f;
        extrapolate<<<weight_blocks, 256>>>(
            x.get(), old.get(), point.get(), momentum, n_spots);
        float point_value = evaluate(point.get(), true);
        CUDA_CHECK(cudaMemset(norm.get(), 0, sizeof(int)));
        projected_norm<<<weight_blocks, 256>>>(
            point.get(), gradient.get(), norm.get(), n_spots);
        int norm_bits;
        CUDA_CHECK(cudaMemcpy(&norm_bits, norm.get(), sizeof(int),
                              cudaMemcpyDeviceToHost));
        float point_norm;
        std::memcpy(&point_norm, &norm_bits, sizeof(float));
        if (point_norm <= gradient_tolerance && !accelerated) {
            converged = true;
            break;
        }

        if (conjugate) {
            float beta = 0.0f;
            if (iterations > 0 && iterations % n_spots != 0) {
                CUDA_CHECK(cudaMemset(model.get(), 0, 2 * sizeof(float)));
                cg_beta_statistics<<<weight_blocks, 256>>>(
                    point.get(), gradient.get(), previous_gradient.get(),
                    model.get(), n_spots);
                float statistics[2];
                CUDA_CHECK(cudaMemcpy(statistics, model.get(), sizeof(statistics),
                                      cudaMemcpyDeviceToHost));
                if (statistics[1] > 1.0e-20f) {
                    beta = std::min(fmaxf(statistics[0] / statistics[1], 0.0f),
                                    10.0f);
                }
            }
            cg_direction<<<weight_blocks, 256>>>(
                point.get(), gradient.get(), previous_gradient.get(),
                direction.get(), beta, n_spots);
            CUDA_CHECK(cudaMemset(model.get(), 0, sizeof(float)));
            direction_slope<<<weight_blocks, 256>>>(
                gradient.get(), direction.get(), model.get(), n_spots);
            if (device_scalar(model.get()) >= 0.0f) {
                cg_direction<<<weight_blocks, 256>>>(
                    point.get(), gradient.get(), previous_gradient.get(),
                    direction.get(), 0.0f, n_spots);
            }
        }

        float trial_step = std::min(step * (conjugate ? 1.5f : 1.1f), 10000.0f);
        float candidate_value = 0.0f;
        bool accepted = false;
        for (int attempt = 0; attempt < 24; ++attempt) {
            if (conjugate) {
                direction_step<<<weight_blocks, 256>>>(
                    point.get(), direction.get(), candidate.get(),
                    trial_step, n_spots);
            } else {
                projected_step<<<weight_blocks, 256>>>(
                    point.get(), gradient.get(), candidate.get(),
                    trial_step, n_spots);
            }
            CUDA_CHECK(cudaMemset(model.get(), 0, 2 * sizeof(float)));
            step_model<<<weight_blocks, 256>>>(
                point.get(), gradient.get(), candidate.get(), model.get(), n_spots);
            float model_host[2];
            CUDA_CHECK(cudaMemcpy(model_host, model.get(), 2 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            candidate_value = evaluate(candidate.get(), false);
            const float limit = conjugate
                ? point_value + 1.0e-4f * model_host[0]
                : point_value + model_host[0]
                    + 0.5f * model_host[1] / trial_step;
            if ((!conjugate || model_host[0] < 0.0f) &&
                candidate_value <= limit + 1.0e-9f) {
                accepted = true;
                break;
            }
            trial_step *= 0.5f;
        }
        if (!accepted) throw std::runtime_error("CUDA line search failed");
        step = trial_step;
        if (accelerated && candidate_value > best_value + 1.0e-9f) {
            // Monotone restart: discard extrapolation, then retry next iteration.
            CUDA_CHECK(cudaMemcpy(old.get(), x.get(), n_spots * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            t = 1.0f;
            continue;
        }
        CUDA_CHECK(cudaMemcpy(old.get(), x.get(), n_spots * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(x.get(), candidate.get(), n_spots * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        best_value = candidate_value;
        t = accelerated ? next_t : 1.0f;
    }

    best_value = evaluate(x.get(), true);
    CUDA_CHECK(cudaMemset(norm.get(), 0, sizeof(int)));
    projected_norm<<<weight_blocks, 256>>>(x.get(), gradient.get(), norm.get(), n_spots);
    int norm_bits;
    CUDA_CHECK(cudaMemcpy(&norm_bits, norm.get(), sizeof(int), cudaMemcpyDeviceToHost));
    float final_norm;
    std::memcpy(&final_norm, &norm_bits, sizeof(float));
    converged = final_norm <= gradient_tolerance;
    IMPTWeightOptimizerResult result;
    result.weights.resize(n_spots);
    CUDA_CHECK(cudaMemcpy(result.weights.data(), x.get(), n_spots * sizeof(float),
                          cudaMemcpyDeviceToHost));
    result.objective = best_value;
    result.projected_gradient = final_norm;
    result.iterations = iterations;
    result.forward_evaluations = forward_evaluations;
    result.gradient_evaluations = gradient_evaluations;
    result.converged = converged;
    return result;
}
