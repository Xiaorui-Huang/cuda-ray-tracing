#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "HitInfo.cuh"
#include "Light.cuh"
#include "Object.cuh"

#include "first_hit.cuh"
#include "generate_ray.cuh"
#include "ray_color.cuh"

#include "util.cuh"

/**
 * @brief Ray trace an image given the Scene
 * 
 * @param camera The camera to use for ray generation
 * @param objects The objects in the scene
 * @param num_objects 
 * @param lights The lights in the scene
 * @param num_lights 
 * @param materials The materials of the objects in the scene
 * @param num_materials
 * @param width The width of the image in pixels
 * @param height The height of the image in pixels
 * @param image - `Output` The image buffer to write to
 */
__global__ void ray_trace_kernel(const Camera &camera,
                                 const Object *objects,
                                 const size_t num_objects,
                                 const Light *lights,
                                 const size_t num_lights,
                                 const Material *materials,
                                 const size_t num_materials,
                                 const size_t width,
                                 const size_t height,
                                 unsigned char *image) {

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index, left to right
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index, top to bottom

    // thread guard
    if (col < width && row < height) {

        // Generate viewing ray
        Ray ray = generate_ray(camera, row, col, width, height);

        HitInfo hit_info;
        if (first_hit(ray, objects, num_objects, 1.0, INFINITY, hit_info)) {
            //defualt debugging color - any error should get black or mild red image as output
            float3d rgb(0.8, 0.3, 0.2);

            // Shoot ray into scene, check for intersection and get color
            rgb = ray_color(ray,
                            1.0,
                            INFINITY,
                            objects,
                            num_objects,
                            lights,
                            num_lights,
                            materials,
                            num_materials);

// debug only check for normal and t calculations
#ifdef DEBUG
// rgb = normal_to_rgb(hit_info.normal);
// rgb = depth_to_rgb(hit_info.t);
#endif

            auto clamp = [](float x) { return x < 0 ? 0 : x > 1 ? 1 : x; };

            // set pixel color to value computed from hit point, light, and n normal image
            image[row * width * 3 + col * 3 + (int)Color::Red] = clamp(rgb.x) * 255.0;
            image[row * width * 3 + col * 3 + (int)Color::Green] = clamp(rgb.y) * 255.0;
            image[row * width * 3 + col * 3 + (int)Color::Blue] = clamp(rgb.z) * 255.0;
        }
    }
}

#endif