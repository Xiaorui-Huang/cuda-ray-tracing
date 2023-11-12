#ifndef HITINFO_CUH
#define HITINFO_CUH

#include "Float3d.cuh"

struct HitInfo {
    float t;   // Distance from the ray origin to the nearest intersection
    float t_far;    // Distance from the ray origin to the farthest intersection
    float3d normal; // Unit normal at the intersection
    int object_index;  // ID of the intersected object (-1 if no intersection)

    __device__ HitInfo() : t(INFINITY), t_far(-INFINITY), normal(float3d()), object_index(-1) {}

    __device__ HitInfo(const float t, const float t_far, const float3d normal, const int object_index)
        : t(t), t_far(t_far), normal(normal), object_index(object_index) {}

    // Additional properties and functions related to hit information can be added here
};

#endif