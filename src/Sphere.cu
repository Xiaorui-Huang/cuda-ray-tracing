#include "Ray.h"
#include "Sphere.cuh"
#include "util.cuh"

/**
 * @brief Sphere intersection test - see Object::intersect()
 */
__device__ bool Sphere::intersect(const Ray &ray,
                                  const float min_t,
                                  const float max_t,
                                  float &t,
                                  float3d &n) const {

    float3d vec = ray.origin - center;

    float a = ray.direction.dot(ray.direction);
    float b = 2 * vec.dot(ray.direction);
    float c = vec.dot(vec) - (radius * radius);
    float t1, t2, t_temp, n_dir = 1;

    /*
    solving                            tex: ||e + td - c|| = r
    equivalently tex: ||origin + t * direction - center || = radius
             tex:  || (origin - center) + t * direction || = radius
                      tex:    || vec + t * direction ||^2 = r^2

    Finally...
    tex: ||direction||^2 * t^2  + 2 * vec.dot(direction) * t + (||vec||^2 - r^2) = 0
    */

    // tex: \| \text {direction} \| ^2 \cdot t ^ 2 + 2 \cdot \text { vec } \cdot \text { direction } \cdot t + (\| \text{vec} \|^2 - r^2) = 0

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;

        if (is_close(discriminant, 0)) {
            t1 = -b / (2 * a);
            if (t1 < min_t || t1 > max_t)
                return false;
            t = t1;
        } else {
            t1 = (-b + sqrt(discriminant)) / (2 * a);
            t2 = (-b - sqrt(discriminant)) / (2 * a);

            if (t1 < min_t && t2 < min_t) // both real roots are invalid
                return false;
            else if (t1 < min_t) { // root t1 is invalid
                t_temp = t2;
                n_dir = -1;          // ray casted from the inside
            } else if (t2 < min_t) { // root t2 is invalid
                t_temp = t1;
                n_dir = -1; // ray casted from the inside
            } else
                t_temp = min(t1, t2);

            if (t_temp > max_t)
                return false;

            t = t_temp;
        }
    }
    // unit normal - reverse the surface normal if ray intersected from the inside
    n = n_dir * (ray.origin + t * ray.direction - center) / radius;
    return true;
}
