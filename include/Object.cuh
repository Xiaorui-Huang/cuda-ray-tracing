#ifndef OBJECT_CUH
#define OBJECT_CUH

#include <math.h>

#include "AABB.cuh"
#include "Plane.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"

#define EPS 1e-4f

enum class ObjectType { AABB, Plane, Sphere, Triangle, TriangleSoup };

union ObjectData {
    AABB aabb;
    Plane plane;
    Sphere sphere;
    Triangle triangle;
};

// Assumes objects, globalTriangleSoup, lights are gloablly variables
struct Object {
    ObjectData data;
    ObjectType type;
    AABB bounding_box;
    int material_index; // Unique identifier for the object

    __host__ __device__ Object(const AABB &aabb)
        : type(ObjectType::AABB), bounding_box(aabb), material_index(-1), data({.aabb = aabb}) {}

    __host__ __device__ Object(const Sphere &sphere)
        : type(ObjectType::Sphere), material_index(-1), data({.sphere = sphere}),
          bounding_box(sphere.center - sphere.radius, sphere.center + sphere.radius) {}

    __host__ __device__ Object(const Triangle &triangle)
        : type(ObjectType::Triangle), material_index(-1), data({.triangle = triangle}),
          bounding_box(triangle.min_corner(), triangle.max_corner()) {}

    __host__ __device__ Object(const Plane &plane)
        : type(ObjectType::Plane), material_index(-1), data({.plane = plane}) {

        // Small epsilon value for the bounding box thickness
        float3d min_bounds(-INFINITY, -INFINITY, -INFINITY);
        float3d max_bounds(INFINITY, INFINITY, INFINITY);

        // Check if the normal is parallel to the x-axis
        if (fabs(plane.normal.x()) > 1.0f - EPS) {
            min_bounds.x() = plane.point.x() - EPS;
            max_bounds.x() = plane.point.x() + EPS;
        }
        // Check if the normal is parallel to the y-axis
        else if (fabs(plane.normal.y()) > 1.0f - EPS) {
            min_bounds.y() = plane.point.y() - EPS;
            max_bounds.y() = plane.point.y() + EPS;
        }
        // Check if the normal is parallel to the z-axis
        else if (fabs(plane.normal.z()) > 1.0f - EPS) {
            min_bounds.z() = plane.point.z() - EPS;
            max_bounds.z() = plane.point.z() + EPS;
        }
        // Set the bounding box for the plane
        bounding_box = AABB(min_bounds, max_bounds);
    }

    __device__ bool intersect(const Ray &ray, float min_t, float max_t, HitInfo &hit_info) const {
        if (!bounding_box.intersect(ray, min_t, max_t, hit_info)) {
            return false; // Bounding box check first for early exit
        }

        // Object-specific intersection
        switch (type) {
        case ObjectType::Plane:
            return data.plane.intersect(ray, min_t, max_t, hit_info);
        case ObjectType::Sphere:
            return data.sphere.intersect(ray, min_t, max_t, hit_info);
        case ObjectType::Triangle:
            return data.triangle.intersect(ray, min_t, max_t, hit_info);
        default:
            return false; // Unrecognized object type
        }
    }
};

#endif