#include "AABB.cuh"
#include "Ray.h"

/**
 * @brief AABB intersection test - see Object::intersect()
 */
__device__ bool
AABB::intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const {

}
