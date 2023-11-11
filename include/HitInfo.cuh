#ifndef HITINFO_CUH
#define HITINFO_CUH

#include "Float3d.cuh"

struct HitInfo {
    float t_near;   // Distance from the ray origin to the nearest intersection
    float t_far;    // Distance from the ray origin to the farthest intersection
    float3d normal; // Normal at the intersection
    int object_id;  // ID of the intersected object (-1 if no intersection)

    __device__ HitInfo() : t_near(INFINITY), t_far(-INFINITY), normal(float3d()), object_id(-1) {}

    __device__ HitInfo(const float t_near, const float t_far, const float3d normal, const int object_id)
        : t_near(t_near), t_far(t_far), normal(normal), object_id(object_id) {}

    // Additional properties and functions related to hit information can be added here
};

#endif