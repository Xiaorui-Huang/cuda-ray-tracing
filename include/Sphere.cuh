#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Vec3d.cuh"

struct Ray;
struct HitInfo;

struct Sphere {
    Vec3d center;
    double radius;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const; 
};

#endif