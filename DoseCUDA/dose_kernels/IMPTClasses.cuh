#ifndef IMPTCLASSES_H
#define IMPTCLASSES_H

#include "CudaClasses.cuh"

#define LUT_LENGTH 400

typedef struct {

    float x;
    float y;
    float mu;
    int energy_id;

} Spot;

typedef struct {

    int spot_start; // Starting spot index
    int n_spots;    // Total
    int energy_id;  // For indexing LUTs

    float r80;
    float energy;

} Layer;

class IMPTBeam : public CudaBeam{

    public:

        struct Model {
            float vsadx;
            float vsady;

            float sourceDistance() const { return (this->vsadx + this->vsady) / 2.0f; }
        } model;    // Beam model parameters

        int n_energies; // Total energies

        Layer * layers; // Layer information
        int n_layers;   // Number of non-empty layers

        Spot * spots;   // All spots, sorted by energy ID
        int n_spots;    // Spot count

        float * divergence_params;  // R80, energy, coefficients
        int dvp_len;    // Length including R80 + energy (stride)

        float * lut_depths;
        float * lut_sigmas;
        float * lut_idds;
        int lut_len;

        __host__ IMPTBeam(IMPTBeam * h_beam);
        __host__ IMPTBeam(float * iso, float gantry_angle, float couch_angle, const Model * model);

        __host__ void importLayers();

        __device__ void interpolateProtonLUT(float wet, float * idd, float * sigma, unsigned layer_id);

        __device__ float sigmaAir(float wet, float distance_to_source, unsigned layer_id);

        __device__ void nuclearHalo(const float wet, float * halo_sigma, float * halo_weight, unsigned layer_id);

        /** @brief Compute the @b squared distance of voxel coordinates to their
         *      nearest point on a pencil beam
         * 	@param spot
         * 		The pencil beam
         * 	@param vox
         * 		The voxel coordinates in BEV
         */
        __device__ float caxDistance(const Spot &spot, const PointXYZ &vox);

};

class IMPTDose : public CudaDose{

    public:

        using CudaDose::CudaDose;

        __host__ IMPTDose(CudaDose * h_dose);

};

__global__ void smoothRayKernel(IMPTDose * dose, CudaBeam * beam, float * SmoothedWETArray);
__global__ void pencilBeamKernel(IMPTDose * dose, IMPTBeam * beam);

void proton_raytrace_cuda(int gpu_id, CudaDose * h_dose, CudaBeam  * h_beam);
void proton_spot_cuda(int gpu_id, IMPTDose * h_dose, IMPTBeam  * h_beam);


#endif