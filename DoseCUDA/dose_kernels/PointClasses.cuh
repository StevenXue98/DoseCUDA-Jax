#ifndef POINT_CLASSES_H
#define POINT_CLASSES_H

#include <cstddef>

typedef struct {

    unsigned i;
    unsigned j;
    unsigned k;

} PointIJK;

typedef struct {

    float x;
    float y;
    float z;

} PointXYZ;

__host__ __device__ static inline float xyz_dotproduct(const PointXYZ &a, const PointXYZ &b) {

    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ static inline PointXYZ xyz_crossproduct(const PointXYZ &a, const PointXYZ &b) {

    PointXYZ res;

    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

#endif /* POINT_CLASSES_H */
