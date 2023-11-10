#ifndef HITINFO_CUH
#define HITINFO_CUH

#include "Float3d.cuh"

struct HitInfo {
    float tNear; // Distance from the ray origin to the nearest intersection
    float tFar;  // Distance from the ray origin to the farthest intersection
    float3d normal; // Normal at the intersection

    // // Constructor to initialize with maximum distance (infinity) and invalid object ID
    // __device__ HitInfo() : tNear(INFINITY), tFar(INFINITY) {
    //     // Normal vector is implicitly initialized (could be zero-initialized or default-constructed
    //     // depending on Vec3)
    // }

    // Additional properties and functions related to hit information can be added here
};

#endif