#ifndef RAY_COLOR_H
#define RAY_COLOR_H

#include "Light.cuh"
#include "Material.h"
#include "Object.cuh"
#include "Ray.h"

#include "blinn_phong_shading.cuh"
#include "first_hit.cuh"

#ifndef MAX_DEPTH
#define MAX_DEPTH 5
#endif

// 🤔🤔🤔 - as reference only
__device__ float3d refract(const float3d &v, const float3d &n, const float &eta) {
    float cos_theta = min(-v.dot(n), 1.0);
    float3d r_out_perp = eta * (v + cos_theta * n);
    float3d r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.dot(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

// Reflect an incoming ray into an out going ray
//
// Inputs:
//   in  incoming _unit_ ray direction
//   n  surface _unit_ normal about which to reflect
// Returns outward _unit_ ray direction

/**
 * @brief Reflect an incoming ray into an out going ray
 * 
 * @param in Incoming `unit` ray direction
 * @param n Surface `unit` normal about which to reflect 
 * @return Outward `unit` ray direction
 */
__device__ float3d reflect(const float3d &in, const float3d &n) { return in - 2 * in.dot(n) * n; }

struct RayState {
    float3d rgb;
    float3d km;
};

/**
 * @brief Calculate the color of a ray
 * 
 * @param ray The ray to calculate the color of
 * @param min_t The minimum t value to consider
 * @param max_t The maximum t value to consider
 * @param objects The objects in the scene
 * @param num_objects 
 * @param lights The lights in the scene
 * @param num_lights 
 * @param materials The materials in the scene
 * @param num_materials 
 * @return float3d The color of the ray
 */
__device__ float3d ray_color(const Ray &ray,
                             const float min_t,
                             const float max_t,
                             const Object *objects,
                             const size_t num_objects,
                             const Light *lights,
                             const size_t num_lights,
                             const Material *materials,
                             const size_t num_materials) {

    float3d rgb = float3d(0);
    RayState ray_stack[MAX_DEPTH];
    unsigned int stack_size = 0;

    float small_t = min_t;

    bool hit = false;
    float3d km;

    // althought the first ray is not a reflection, we use the name for later reflection semantics
    Ray reflected_ray = ray;
    HitInfo hit_info;

    for (int i = 0; i < MAX_DEPTH; i++) {
        if (!first_hit(reflected_ray, objects, num_objects, small_t, max_t, hit_info))
            break;
        hit = true;
        //since reflected ray is not as close to the camera as the original ray,
        //but we still need to set a small t to avoid self-intersection
        small_t = 0.001f;

        km = materials[objects[hit_info.object_index].material_index].km;
        ray_stack[stack_size++] = {blinn_phong_shading(reflected_ray,
                                                       hit_info,
                                                       objects,
                                                       num_objects,
                                                       lights,
                                                       num_lights,
                                                       materials,
                                                       num_materials),
                                   km};

        reflected_ray.origin = reflected_ray.origin + hit_info.t * reflected_ray.direction;
        reflected_ray.direction =
            reflect(reflected_ray.direction.normalized(), hit_info.normal.normalized());
    }
    if (hit) {
        rgb = ray_stack[stack_size - 1].rgb;
        for (int i = stack_size - 2; i >= 0; i--)
            rgb = rgb * ray_stack[i].km + ray_stack[i].rgb;
    }
    return rgb;
}

#endif