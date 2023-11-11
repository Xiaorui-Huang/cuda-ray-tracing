#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "HitInfo.cuh"
#include "Light.cuh"
#include "Object.cuh"

#include "first_hit.cuh"
#include "generate_ray.cuh"

#include "util.cuh"


//  * @brief Ray Trace an image given the Scene

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
 * @param rays - `Memory`
 * @param hit_infos - `Memory`
 * @param image - `Output`
 */
__global__ void ray_trace_kernel(const Camera &camera,
                                 const Object *objects,
                                 const size_t num_objects,
                                 const Light *lights,
                                 const size_t num_lights,
                                 const size_t width,
                                 const size_t height,
                                 Ray *rays,
                                 HitInfo *hit_infos,
                                 unsigned char *image) {

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index, left to right
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index, top to bottom

    // thread guard
    if (col < width && row < height) {
        int flat_idx = row * width + col;

        // Generate viewing ray
        Ray ray;
        generate_ray(camera, row, col, width, height, ray);
        if (row == 0) {
            printf("ray direction: %f %f %f\n",
                   ray.direction.x(),
                   ray.direction.y(),
                   ray.direction.z());
        }

        float3d rgb(0.8, 0.3, 0.2);

        // since these are local variables to the kernel, we need to copy to global mem to pass into first hit kernel

        int grid_size = 16 * 16;
        int block_size = (num_objects + grid_size - 1) / grid_size;

        first_hit<<<block_size, block_size, sizeof(HitInfo) * block_size>>>(
            rays[flat_idx], objects, num_objects, 1.0, INFINITY, hit_infos);
        
        // iterate hit_infos over each block to find the final first hit 
        
        if (row == 0) {
            printf("hit info normal: %f %f %f %d %f\n",
                   hit_infos[flat_idx].normal.x(),
                   hit_infos[flat_idx].normal.y(),
                   hit_infos[flat_idx].normal.z(),
                   hit_infos[flat_idx].object_id,
                   hit_infos[flat_idx].t_near);

        }
        if(hit_infos[flat_idx].object_id == -1) {
            return;
        }

        // Shoot ray into scene, check for intersection and get color
        // ray_color(ray, objects, num_objects, lights, num_lights, image, row, col, width, height, rgb);

        auto clamp = [](float x) { return x < 0 ? 0 : x > 1 ? 1 : x; };

        // set pixel color to value computed from hit point, light, and n normal image
        auto normal_to_rgb = [](const float3d &n, float3d &rgb) {
            rgb.x() = 255.0 * (n.x() * 0.5 + 0.5);
            rgb.y() = 255.0 * (n.y() * 0.5 + 0.5);
            rgb.z() = 255.0 * (n.z() * 0.5 + 0.5);
        };

        normal_to_rgb(hit_infos[flat_idx].normal, rgb);

        image[row * width * 3 + col * 3 + (int)Color::Red] = clamp(rgb.x()) * 255.0;
        image[row * width * 3 + col * 3 + (int)Color::Green] = clamp(rgb.y()) * 255.0;
        image[row * width * 3 + col * 3 + (int)Color::Blue] = clamp(rgb.z()) * 255.0;
    }
}

#endif