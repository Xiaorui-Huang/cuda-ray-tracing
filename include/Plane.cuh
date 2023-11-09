#ifndef PLANE_CUH
#define PLANE_CUH

#include "Vec3d.cuh"

struct Ray;
struct HitInfo;

struct Plane {
    Vec3d point;
    Vec3d normal;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

# endif