#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "HitInfo.cuh"
#include "Light.cuh"
#include "Object.cuh"

#include "first_hit.cuh"
#include "generate_ray.cuh"

#include "util.h"

/**
 * @brief Ray Trace an image given the Scene
 *
 * @param camera
 * @param objects
 * @param num_objects
 * @param lights
 * @param num_lights
 * @param width
 * @param height
 * @param image
 */
__global__ void ray_trace_kernel(const Camera &camera,
                                 const Object *objects,
                                 const size_t num_objects,
                                 const Light *lights,
                                 const size_t num_lights,
                                 const size_t width,
                                 const size_t height,
                                 unsigned char *image) {

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index, left to right
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index, top to bottom

    // thread guard
    if (col < width && row < height) {

        // Generate viewing ray
        Ray ray;
        generate_ray(camera, row, col, width, height, ray);
        if (row == 0) {
            printf("ray direction: %f %f %f\n", ray.direction.x(), ray.direction.y(),
                   ray.direction.z());
        }

        float3d rgb(0.8, 0.3, 0.2);

        // since these are local variables to the kernel, we need to copy to global mem to pass into first hit kernel

        // HitInfo *d_hit_info = (HitInfo *)malloc(sizeof(HitInfo));
        // Ray *d_ray = (Ray *)malloc(sizeof(Ray));
        // memcpy(d_ray, &ray, sizeof(Ray));
        // *d_ray = ray;

        // int grid_size = 16 * 16;
        // int block_size = (num_objects + grid_size - 1) / grid_size;

        // first_hit<<<block_size, block_size, sizeof(HitInfo) * block_size>>>(
        //     *d_ray, objects, num_objects, 1.0, INFINITY, *d_hit_info);

        // HitInfo hit_info = *d_hit_info;

        // if (d_hit_info->object_id != 0) {
        //     printf("hit object id: %d\n", d_hit_info->object_id);
        // }

        // Shoot ray into scene, check for intersection and get color
        // ray_color(ray, objects, num_objects, lights, num_lights, image, row, col, width, height, rgb);

        auto clamp = [](float x) { return x < 0 ? 0 : x > 1 ? 1 : x; };

        // set pixel color to value computed from hit point, light, and
        // n normal image
        // auto normal_to_rgb = [](const float3d &n, float3d &rgb) {
        //     rgb.x() = 255.0 * (n(0) * 0.5 + 0.5);
        //     rgb.y() = 255.0 * (n(1) * 0.5 + 0.5);
        //     rgb.z() = 255.0 * (n(2) * 0.5 + 0.5);
        // };

        // normal_to_rgb(d_hit_info->normal, rgb);

        image[row * width * 3 + col * 3 + (int)Color::Red] = clamp(rgb.x()) * 255.0;
        image[row * width * 3 + col * 3 + (int)Color::Green] = clamp(rgb.y()) * 255.0;
        image[row * width * 3 + col * 3 + (int)Color::Blue] = clamp(rgb.z()) * 255.0;
    }
}

#endif