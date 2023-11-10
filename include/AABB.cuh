#ifndef AABB_CUH
#define AABB_CUH

#include "Float3d.cuh"

struct Ray;
struct HitInfo;

struct AABB {
    float3d min; // Minimum point
    float3d max; // Maximum point

    __host__ __device__ AABB(): min(INFINITY), max(-INFINITY) {}
    
    __host__ __device__ AABB(const float3d &_min, const float3d &_max): min(_min), max(_max) {}
    
    __device__ float3d center() const { return (min + max) * 0.5; }

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

struct BVHNode {
    AABB box;
    int leftIndex;   // Index of the left child in the BVH node array
    int rightIndex;  // Index of the right child in the BVH node array
    int objectIndex; // Index of the associated object, -1 if it's not a leaf node
};
```

#endif