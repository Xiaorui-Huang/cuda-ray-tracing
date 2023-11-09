#ifndef TRIANGLE_SOUP_CUH
#define TRIANGLE_SOUP_CUH

#include "Vec3d.cuh"

struct Ray;
struct HitInfo;

struct TriangleSoup {
    int startOffset;
    int count;
    int materialIndex;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

#endif