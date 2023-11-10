#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "Vec3d.cuh"
#include "util.h"

struct Ray;
struct HitInfo;

struct Triangle {
    Vec3d corners[3];

    __host__ __device__ Triangle(Vec3d a, Vec3d b, Vec3d c) : corners{a, b, c} {}
    
    __host__ __device__ Vec3d minCorner() const {
        Vec3d minCorner = corners[0];
        for (int i = 1; i < 3; i++) {
            minCorner.x() = min(corners[i].x(), minCorner.x());
            minCorner.y() = min(corners[i].y(), minCorner.y());
            minCorner.z() = min(corners[i].z(), minCorner.z());
        }
        return minCorner;
    }

    __host__ __device__ Vec3d maxCorner() const {
        Vec3d maxCorner = corners[0];
        for (int i = 1; i < 3; i++) {
            maxCorner.x() = max(corners[i].x(), maxCorner.x());
            maxCorner.y() = max(corners[i].y(), maxCorner.y());
            maxCorner.z() = max(corners[i].z(), maxCorner.z());
        }
        return maxCorner;
    }

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const;
};

#endif