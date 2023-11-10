#ifndef PLANE_CUH
#define PLANE_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct Plane {
    float3d point;
    float3d normal;

    __device__ bool intersect(const Ray &ray, float min_t, float max_t, HitInfo &hit_info) const;
};

# endif