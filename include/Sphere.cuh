#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct Sphere {
    float3d center;
    float radius;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const; 
};

#endif