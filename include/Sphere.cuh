#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct Sphere {
    float3d center;
    float radius;

    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const;
};

#endif
