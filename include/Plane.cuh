#ifndef PLANE_CUH
#define PLANE_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct Plane {
    float3d point;
    float3d normal;

    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const;
};

#endif