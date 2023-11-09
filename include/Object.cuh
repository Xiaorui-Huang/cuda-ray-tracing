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

    // define defalt constructors, since union have non-trivial members
    __host__ __device__ ObjectData() { memset(this, 0, sizeof(ObjectData)); }
};

// Assumes objects, globalTriangleSoup, lights are gloablly variables
struct Object {
    ObjectData data;
    ObjectType type;
    AABB boundingBox;
    int materialIndex; // Unique identifier for the object

    __host__ __device__ Object(AABB aabb)
        : type(ObjectType::AABB), boundingBox(aabb), materialIndex(-1) {
        data.aabb = aabb;
    }

    __host__ __device__ Object(Plane plane)
        : type(ObjectType::Plane), materialIndex(-1) {
        data.plane = plane;

        // Small epsilon value for the bounding box thickness
        Vec3d minBounds(-INFINITY, -INFINITY, -INFINITY);
        Vec3d maxBounds(INFINITY, INFINITY, INFINITY);

        // Check if the normal is parallel to the x-axis
        if (fabs(plane.normal.x()) > 1.0f - EPS) {
            minBounds.x() = plane.point.x() - EPS;
            maxBounds.x() = plane.point.x() + EPS;
        }
        // Check if the normal is parallel to the y-axis
        else if (fabs(plane.normal.y()) > 1.0f - EPS) {
            minBounds.y() = plane.point.y() - EPS;
            maxBounds.y() = plane.point.y() + EPS;
        }
        // Check if the normal is parallel to the z-axis
        else if (fabs(plane.normal.z()) > 1.0f - EPS) {
            minBounds.z() = plane.point.z() - EPS;
            maxBounds.z() = plane.point.z() + EPS;
        }
        // Set the bounding box for the plane
        boundingBox = AABB(minBounds, maxBounds);
    }

    __host__ __device__ Object(Sphere sphere)
        : type(ObjectType::Sphere), materialIndex(-1) {
        data.sphere = sphere;
        boundingBox = AABB(sphere.center - sphere.radius, sphere.center + sphere.radius);
    }

    __host__ __device__ Object(Triangle triangle)
        : type(ObjectType::Triangle), materialIndex(-1) {
        data.triangle = triangle;
        boundingBox = AABB(triangle.minCorner(), triangle.maxCorner());
    }

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const {
        if (!boundingBox.intersect(ray, minT, maxT, hitInfo)) {
            return false; // Bounding box check first for early exit
        }

        // Object-specific intersection
        switch (type) {
        case ObjectType::Plane:
            return data.plane.intersect(ray, minT, maxT, hitInfo);
        case ObjectType::Sphere:
            return data.sphere.intersect(ray, minT, maxT, hitInfo);
        case ObjectType::Triangle:
            return data.triangle.intersect(ray, minT, maxT, hitInfo);
        default:
            return false; // Unrecognized object type
        }
    }
};

#endif