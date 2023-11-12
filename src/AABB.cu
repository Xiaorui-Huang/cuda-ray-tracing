#include "AABB.cuh"
#include "Ray.h"

#include <cassert>
/**
 * @brief AABB intersection test - see [[include/Object.cuh]]::intersect()
 */
__device__ bool
AABB::intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const {
    assert(false); // TODO: Implement this
}
