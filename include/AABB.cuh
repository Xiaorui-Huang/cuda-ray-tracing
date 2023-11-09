#ifndef AABB_CUH
#define AABB_CUH

#include "Vec3d.cuh"

struct Ray;
struct HitInfo;

struct AABB {
    Vec3d min; // Minimum point
    Vec3d max; // Maximum point

    __host__ __device__ AABB(): min(INFINITY), max(-INFINITY) {}
    
    __host__ __device__ AABB(const Vec3d &_min, const Vec3d &_max): min(_min), max(_max) {}
    
    __device__ Vec3d center() const { return (min + max) * 0.5; }

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

#endif