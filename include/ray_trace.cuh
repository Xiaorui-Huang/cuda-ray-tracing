#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "Object.cuh"
#include "Light.cuh"
#include "generate_ray.cuh"

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
    if (col < width && row < height) {

        Ray ray;
        generate_ray(camera, row, col, width, height, ray);
        
        float3d rgb;
        // ray_color(ray, objects, num_objects, lights, num_lights, image, row, col, width, height, rgb);
    }
}

#endif