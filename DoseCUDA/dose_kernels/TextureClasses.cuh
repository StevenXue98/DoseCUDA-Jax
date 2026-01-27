#pragma once

#ifndef TEXTURECLASSES_H
#define TEXTURECLASSES_H

#include <cuda_runtime.h>
#include "PointClasses.cuh"


class Texture3D {
    cudaArray_t         m_arr;
    cudaTextureObject_t m_tex;
    bool                m_isCopy;

    void makeArray(const PointIJK &size);
    void makeTexture(cudaTextureFilterMode filterMode, float border);

    Texture3D(const float data[], const PointIJK &size, cudaTextureFilterMode filterMode, float border, cudaMemcpyKind direction);

public:
    Texture3D(const Texture3D &other):
        m_arr(other.m_arr), m_tex(other.m_tex), m_isCopy(true) { }

    Texture3D(Texture3D &&other):
        m_arr(other.m_arr), m_tex(other.m_tex), m_isCopy(other.m_isCopy)
    {
        other.m_arr    = NULL;
        other.m_tex    = 0;
        other.m_isCopy = true;
    }

    ~Texture3D();

    /** @brief Create a texture from 3D host data
     *  @param h_data
     *      A pointer to an array of host data
     *  @param size
     *      Voxel dimensions of the host array
     *  @param filterMode
     *      Desired filtering mode for sampling from this texture
     *  @returns A new 3D texture to be used for sampling in device code
     */
    static __host__ Texture3D fromHostData(const float h_data[], const PointIJK &size, cudaTextureFilterMode filterMode=cudaFilterModePoint, float border=0.0f) {

        return Texture3D(h_data, size, filterMode, border, cudaMemcpyHostToDevice);
    }

    /** @brief Create a texture from 3D device data
     *  @param h_data
     *      A pointer to an array of device data
     *  @param size
     *      Voxel dimensions of the device array
     *  @param filterMode
     *      Desired filtering mode for sampling from this texture
     *  @returns A new 3D texture to be used for sampling in device code
     */
    static __host__ Texture3D fromDeviceData(const float d_data[], const PointIJK &size, cudaTextureFilterMode filterMode=cudaFilterModePoint, float border=0.0f) {

        return Texture3D(d_data, size, filterMode, border, cudaMemcpyDeviceToDevice);
    }

    /** @brief Sample from this texture using real-valued pixel coordinates */
    __device__ float sample(float x, float y, float z) const {

        return tex3D<float>(m_tex, x, y, z);
    }

    /** @brief Sample from this texture using a coordinate struct */
    __device__ float sample(const PointXYZ &xyz) const {

        return sample(xyz.x, xyz.y, xyz.z);
    }
};


#endif /* TEXTURECLASSES_H */
