#ifndef CUDA_CLASSES_H
#define CUDA_CLASSES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math_constants.h>
#include "PointClasses.cuh"
#include "TextureClasses.cuh"

#define TILE_WIDTH 4

static inline void cuda_check(cudaError_t err)
{
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(err) cuda_check(err)


class CudaBeam{

    public:

        PointXYZ iso;
        PointXYZ src;
        float gantry_angle;
        float couch_angle;

        float singa, cosga; // Cached gantry angle trig functions
        float sinta, costa; // Cached couch angle trig functions

        __host__ CudaBeam(CudaBeam * h_beam);
        __host__ CudaBeam(float * iso, float gantry_angle, float couch_angle, float src_dist);

        __device__ void unitVectorToSource(const PointXYZ * point_xyz, PointXYZ * uvec) {

            uvec->x = this->src.x - point_xyz->x;
            uvec->y = this->src.y - point_xyz->y;
            uvec->z = this->src.z - point_xyz->z;

            const auto norm = rnorm3df(uvec->x, uvec->y, uvec->z);
            uvec->x *= norm;
            uvec->y *= norm;
            uvec->z *= norm;

        }

        __device__ float distanceToSource(const PointXYZ * point_xyz);

        __device__ void pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head);

        __device__ void pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img);

        __device__ void pointXYZClosestCAXPoint(const PointXYZ * point_xyz, PointXYZ * point_cax);

        __device__ float pointXYZDistanceToCAX(const PointXYZ * point_head_xyz);

        __device__ float pointXYZDistanceToSource(const PointXYZ * point_img_xyz);

};

class CudaDose{

    public:

        PointIJK img_sz;
        unsigned int num_voxels;

        float spacing;

        float * DoseArray;
        float * DensityArray;
        float * WETArray;

        __host__ CudaDose(CudaDose * h_dose);
        __host__ CudaDose(size_t img_sz[], float spacing);

        __device__ bool pointIJKWithinImage(const PointIJK * point_ijk) {
            return point_ijk->i < this->img_sz.i
                && point_ijk->j < this->img_sz.j
                && point_ijk->k < this->img_sz.k;
        }

        __device__ bool textureXYZWithinImage(const PointXYZ * point_xyz) {
            return point_xyz->x >= 0.0f && point_xyz->x < (float)this->img_sz.i
                && point_xyz->y >= 0.0f && point_xyz->y < (float)this->img_sz.j
                && point_xyz->z >= 0.0f && point_xyz->z < (float)this->img_sz.k;
        }

        __device__ unsigned int pointIJKtoIndex(const PointIJK * point_ijk) {
            return point_ijk->i + this->img_sz.i * (point_ijk->j + this->img_sz.j * point_ijk->k);
        }

        __device__ void pointXYZtoTextureXYZ(const PointXYZ * point_xyz, PointXYZ * tex_xyz, CudaBeam * beam) {
            tex_xyz->x = fmaf(1.0f / this->spacing, point_xyz->x + beam->iso.x, 0.5f);
            tex_xyz->y = fmaf(1.0f / this->spacing, point_xyz->y + beam->iso.y, 0.5f);
            tex_xyz->z = fmaf(1.0f / this->spacing, point_xyz->z + beam->iso.z, 0.5f);
        }

        __device__ void pointIJKtoXYZ(const PointIJK * point_ijk, PointXYZ * point_xyz, CudaBeam * beam) {
            point_xyz->x = (float)point_ijk->i * this->spacing - beam->iso.x;
	        point_xyz->y = (float)point_ijk->j * this->spacing - beam->iso.y;
	        point_xyz->z = (float)point_ijk->k * this->spacing - beam->iso.z;
        }

        __device__ void pointXYZtoIJK(const PointXYZ * point_xyz, PointIJK * point_ijk, CudaBeam * beam) {
            point_ijk->i = (int)roundf((point_xyz->x + beam->iso.x) / this->spacing);
            point_ijk->j = (int)roundf((point_xyz->y + beam->iso.y) / this->spacing);
            point_ijk->k = (int)roundf((point_xyz->z + beam->iso.z) / this->spacing);
        }

};

__global__ void rayTraceKernel(CudaDose * dose, CudaBeam * beam, Texture3D DensityTexture);

/** @brief Binary search to find the first instance of a key in a sorted array,
 *      or alternatively, where such a key should be inserted to maintain sorted
 *      ordering
 *  @see https://docs.python.org/3/library/bisect.html#examples
 *      for motivating examples and see
 *      https://en.cppreference.com/w/cpp/algorithm/lower_bound
 *      for the naming convention
 */
template <class T>
__host__ __device__ static inline int lowerBound(const T data[], int len, T key) {

    int l = 0, r = len;

    while (l < r) {
        int mid = (l + r) / 2;
        if (key <= data[mid]) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

__device__ static inline float sqr(float x)
{
    return x * x;
}

__device__ static inline float clamp(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

#endif