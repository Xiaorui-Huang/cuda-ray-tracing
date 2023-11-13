#ifndef OBJECT_CUH
#define OBJECT_CUH

#include <math.h>

#include "AABB.cuh"

#include "Plane.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"
#include <bvh/BVH.cuh>

#define EPS 1e-4f

enum class ObjectType { BVH, Plane, Sphere, Triangle, TriangleSoup };

union ObjectData {
    BVHNode bvh;
    Plane plane;
    Sphere sphere;
    Triangle triangle;
};

struct Object {
    ObjectType type;
    ObjectData data;
    AABB box;
    int material_index; // Unique identifier for the object

    // Object(const BVHNode &bvh)
    //     : type(ObjectType::BVH), material_index(-1), data({.bvh = bvh}),
    //     box

    __host__ __device__ Object(const Sphere &sphere)
        : type(ObjectType::Sphere), material_index(-1), data({.sphere = sphere}),
          box(sphere.center - sphere.radius, sphere.center + sphere.radius) {}

    __host__ __device__ Object(const Triangle &triangle)
        : type(ObjectType::Triangle), material_index(-1), data({.triangle = triangle}),
          box(triangle.min_corner(), triangle.max_corner()) {}

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
        box = AABB(min_bounds, max_bounds);
    }

    /**
     * @brief Test if a ray intersects with `this` object. `n` and `t` are only set if the ray intersects with the object.
     * 
     * @param ray The ray to test for intersection 
     * @param min_t The minimum distance to consider for intersection
     * @param max_t The maximum distance to consider for intersection 
     * @param t `Output` parameter for the distance to the point of intersection
     * @param n `Output` unit normal vector to the point of intersection 
     * @return true if the ray intersects with the object, false otherwise
     */
    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const;
};

__device__ bool Object::intersect(const Ray &ray,
                                  const float min_t,
                                  const float max_t,
                                  float &t,
                                  float3d &n) const {
    // Object-specific intersection
    switch (type) {
    case ObjectType::BVH:
        return data.bvh.intersect(ray, min_t, max_t, t, n);
    case ObjectType::Plane:
        return data.plane.intersect(ray, min_t, max_t, t, n);
    case ObjectType::Sphere:
        return data.sphere.intersect(ray, min_t, max_t, t, n);
    case ObjectType::Triangle:
        return data.triangle.intersect(ray, min_t, max_t, t, n);
    default:
        return false; // Unrecognized object type
    }
}
/*
   Use case for max_t

Shadow Rays:

In ray tracing, shadow rays are cast from the point of intersection to each
light source to determine visibility (i.e., whether the point is in shadow). If
an intersection with another object occurs at a distance less than the distance
to the light source, the point is in shadow. Here, max_t would be set to the
distance to the light source. Any intersection beyond this distance is
irrelevant for shadow calculation.

   Depth of Field and Focus:

For simulating depth of field, intersections might only be considered within a
certain range of distances from the camera that align with the camera's focus.
Here, max_t can define the far limit of this in-focus range. Optimization in
Scene Hierarchy Traversal:

In spatial data structures like BVH (Bounding Volume Hierarchies) or Octrees
used for efficient ray tracing, max_t can be used to skip entire sections of the
hierarchy. If the closest intersection found so far is closer than max_t, any
node or bounding volume further than max_t can be safely ignored. Reflection and
Refraction Rays:

For reflection and refraction rays, max_t can limit how far these rays travel,
which can be used to simulate effects like reflective or refractive surfaces
fading out over distance.

*/

#endif