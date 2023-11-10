#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct Sphere {
    float3d center;
    float radius;

    __device__ bool intersect(const Ray &ray, float min_t, float max_t, HitInfo &hit_info) const; 
};

#endif