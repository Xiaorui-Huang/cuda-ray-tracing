#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "Vec3d.cuh"

struct Ray;
struct HitInfo;

struct Triangle {
    Vec3d corners[3];
    int materialIndex;
    int soupID;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

#endif