#include "Plane.cuh"
#include "Ray.h"

/**
 * @brief Plane intersection test - see Object::intersect()
 */
__device__ bool
Plane::intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const {
    // t = (p - e).n / d.n
    float sol = (point - ray.origin).dot(normal) / (ray.direction.dot(normal));
    if (sol < min_t || sol > max_t)
        return false;
    t = sol;

    // n = p/|p|
    n = normal.normalized();
    return true;
}
