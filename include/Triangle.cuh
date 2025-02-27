#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "Float3d.cuh"
#include "util.cuh"

struct Ray;
struct HitInfo;

struct Triangle {
    // The three corners of the triangle
    float3d corners[3];

    __host__ __device__ Triangle(float3d a, float3d b, float3d c) : corners{a, b, c} {}
    
    __host__ __device__ float3d min_corner() const {
        float3d corner = corners[0];
        for (int i = 1; i < 3; i++) {
            corner.x = min(corners[i].x, corner.x);
            corner.y = min(corners[i].y, corner.y);
            corner.z = min(corners[i].z, corner.z);
        }
        return corner;
    }

    __host__ __device__ float3d max_corner() const {
        float3d corner = corners[0];
        for (int i = 1; i < 3; i++) {
            corner.x = max(corners[i].x, corner.x);
            corner.y = max(corners[i].y, corner.y);
            corner.z = max(corners[i].z, corner.z);
        }
        return corner;
    }

    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const;
};

#endif