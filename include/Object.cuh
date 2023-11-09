#ifndef OBJECT_CUH
#define OBJECT_CUH

#include "AABB.cuh"
#include "Plane.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"
#include "TriangleSoup.cuh"

enum class ObjectType { AABB, Plane, Sphere, Triangle, TriangleSoup };

union ObjectData {
    AABB aabb;
    Plane plane;
    Sphere sphere;
    Triangle triangle;
    TriangleSoup triangleSoup;

    // define defalt constructors, since union have non-trivial members
    __host__ __device__ ObjectData() { memset(this, 0, sizeof(ObjectData)); }
};

// Assumes objects, globalTriangleSoup, lights are gloablly variables
struct Object {
    ObjectData data;
    ObjectType type;
    AABB boundingBox;
    int materialIndex; // Unique identifier for the object

    __host__ __device__ Object(AABB aabb, int index)
        : type(ObjectType::AABB), boundingBox(aabb), materialIndex(index) {
        data.aabb = aabb;
    }

    __host__ __device__ Object(Plane plane, int index)
        : type(ObjectType::Plane), materialIndex(index) {
        data.plane = plane;
        boundingBox = AABB(-INFINITY, INFINITY);
    }

    __host__ __device__ Object(Sphere sphere, int index)
        : type(ObjectType::Sphere), materialIndex(index) {
        data.sphere = sphere;
        boundingBox = AABB(sphere.center - sphere.radius, sphere.center + sphere.radius);
    }

    __host__ __device__ Object(Triangle triangle, int index)
        : type(ObjectType::Triangle), materialIndex(index) {
        data.triangle = triangle;
        // boundingBox = AABB(triangle.min, triangle.max);
    }

    // We use default/empty Bounding Box for TriangleSoup
    // Box will never be hit.
    // Instead every Triangle will be 
    __host__ __device__ Object(TriangleSoup triangleSoup, int index)
        : type(ObjectType::TriangleSoup), materialIndex(index), boundingBox() {
        data.triangleSoup = triangleSoup;
    }

    __host__

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
        case ObjectType::TriangleSoup:
            return data.triangleSoup.intersect(ray, minT, maxT, hitInfo);
        default:
            return false; // Unrecognized object type
        }
    }
};

#endif