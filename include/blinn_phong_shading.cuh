#ifndef BLINN_PHONG_CUH
#define BLINN_PHONG_CUH

// #include "Float3d.cuh"
#include "HitInfo.cuh"
#include "Light.cuh"
#include "Material.h"
#include "Object.cuh"
#include "Ray.h"

#include "first_hit.cuh"

__device__ float3d blinn_phong_shading(const Ray &ray,
                                       const HitInfo &hit_info,
                                       const Object *objects,
                                       const size_t num_objects,
                                       const Light *lights,
                                       const size_t num_lights,
                                       const Material *materials,
                                       const size_t num_materials) {
    Ray shadow_ray; // ray for testing shadows
    HitInfo shadow_info;
    float max_t, epsilon = 0.0001f;
    Material material = materials[objects[hit_info.object_index].material_index];

    // Hardcoded ambient light intensity;
    float3d Ia = float3d(0.1f);
    float3d rgb = Ia * material.ka;
    float3d half_vec;
    float3d normal = hit_info.normal;


    for (size_t i = 0; i < num_lights; i++) {
        Light light = lights[i];
        shadow_ray.origin = ray.origin + hit_info.t * ray.direction;

        // sets shadow ray direction and max_t to consider based on light position
        light.direction(shadow_ray.origin, shadow_ray.direction, max_t);

        if (!first_hit(shadow_ray, objects, num_objects, epsilon, max_t, shadow_info)) {

            // Diffuse reflection - Lambertian shading
            rgb += max(0.0f, normal.dot(shadow_ray.direction.normalized())) *
                   (light.color * material.kd);

            // specular reflection - Blinn-Phong reflection model

            // 4.5.2 page 82 on Fundamentals of Computer Graphics
            // l + v, where
            //  - l is the direction of the shadow_ray, and
            //  - v is the opposite direction of the ray's direction
            //    (pointing to the camera instead of pointing to the point of intersection)

            // half_vec = (shadow_ray.direction.normalized() - ray.direction.normalized()).normalized();
            half_vec = (shadow_ray.direction + -ray.direction).normalized();
            // [max(0, n.half)]^p * (I ∘ ks)
            rgb += pow(max(0.0f, normal.dot(half_vec)), material.phong_exponent) *
                   (light.color * material.ks);
        }
    }
    return rgb;
}

#endif