#include "Ray.h"
#include "Triangle.cuh"
#include "util.cuh"

/**
  * @brief Triangle intersection test - see [[include/Object.cuh]]::intersect()
 */
__device__ bool Triangle::intersect(const Ray &ray,
                                    const float min_t,
                                    const float max_t,
                                    float &t,
                                    float3d &n) const {

    // Extract vertices of the triangle
    float3d a = corners[0];
    float3d b = corners[1];
    float3d c = corners[2];

    // Edge vectors of the triangle
    float3d a_b = b - a;
    float3d a_c = c - a;

    // Ray direction and origin
    float3d d = ray.direction;
    float3d e = ray.origin;

    // Matrix determinant explanation:
    // A = | a  b  c |
    //     | d  e  f |
    //     | g  h  i |
    //
    // v1 = [a, b, c]
    // v2 = [d, e, f]
    // v3 = [g, h, i]
    //
    // det(A) = v1 . (v2 x v3)
    // Here, v1 = a_b, v2 = a_c, v3 = d (ray direction)
    // Compute normal to the plane of the triangle

    float3d plane_normal = d.cross(a_c);
    float det = a_b.dot(plane_normal);

    // If det is near zero, ray lies in the plane of the triangle
    if (is_close(det, 0.0f))
        return false;

    float inv_det = 1.0f / det;

    // Calculate distance from vertex a to ray origin
    float3d a_e = e - a;

    // Calculate u parameter using dot product
    // The u_parameter is a barycentric coordinate, representing the intersection point's
    // position relative to triangle's edge AB. The dot product of vector a_e (from vertex A to ray origin)
    // and plane_normal (perpendicular to the triangle) measures the projection of a_e onto the plane normal.
    // This projection, scaled by the inverse determinant (inv_det), correlates to the barycentric coordinate u.
    // If u is not between 0 and 1, the intersection point is outside the triangle.
    float u_parameter = a_e.dot(plane_normal) * inv_det;
    if (u_parameter < 0.0f || u_parameter > 1.0f)
        return false;

    // Prepare to test v parameter
    float3d q_vector = a_e.cross(a_b);

    // Calculate v parameter and test bound
    float v_parameter = d.dot(q_vector) * inv_det;
    if (v_parameter < 0.0f || u_parameter + v_parameter > 1.0f)
        return false;

    // Calculate t, ray intersects triangle
    t = a_c.dot(q_vector) * inv_det;

    // Check if intersection is within the bounds of the ray
    if (t < min_t || t > max_t)
        return false;

    // Compute the normal at the intersection
    n = a_b.cross(a_c).normalized();

    return true;
}
