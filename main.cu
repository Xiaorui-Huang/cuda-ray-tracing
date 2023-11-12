#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "Float3d.cuh"
#include "Light.cuh"
#include "Object.cuh"

#include "Camera.h"
#include "Material.h"

#include "ray_trace.cuh"

#include "read_json.h"
#include "write_ppm.h"

#include <cuda_runtime.h>
#define N 32

/**
 * Converts the given position to one which is relative to this context.
 *
 * ```ts
 * let i: number = 1;
 * const a = true;
 * ```
 *
 * @param value The position to convert.
 *
 * @returns The position in local space.
 */
__global__ void show_material(Material *d_materials) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Material: %f\n", d_materials[i].phong_exponent);
}

int main(int argc, char *argv[]) {
    Camera camera;
    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // Read a camera and scene description from given .json file
    unsigned int width = 640;
    unsigned int height = 360;

    readJson(argc <= 1 ? "../data/bunny.json" : argv[1], camera, objects, lights, materials);
    // readJson(argc <= 1 ? "../data/inside-a-sphere.json" : argv[1], camera, objects, lights, materials);

    std::vector<unsigned char> h_rgb_image(3 * width * height);
    size_t image_size = h_rgb_image.size() * sizeof(unsigned char);

    Camera *d_camera;           // ptr to camera
    Object *d_objects;          // array of objects
    Material *d_materials;      // array of materials
    Light *d_lights;            // array of lights
    unsigned char *d_rgb_image; // array of rgb image pixels

    // send data to GPU
    // very well suited for __constant__ use case, except the data is too large
    to_cuda(d_camera, &camera);
    to_cuda(d_objects, objects.data(), objects.size());
    to_cuda(d_lights, lights.data(), lights.size());
    to_cuda(d_materials, materials.data(), materials.size());
    // Good use for 2D or 3D array, but accessing is too complicated
    // https: //docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cudaMallocPitched#device-memory
    cudaMalloc(&d_rgb_image, image_size);

    std::cout << "Memory Cost: "
              << sizeof(Camera) + objects.size() * sizeof(Object) +
                     materials.size() * sizeof(Material) + lights.size() * sizeof(Light)
              << std::endl;

    dim3 block_per_grid(N, N);
    // To ensure all pixels are processed, we round up the number of blocks (reduces occupancy - i.e. empty threads)
    dim3 thread_per_block(ceil(width, block_per_grid.x), ceil(height, block_per_grid.y));

    // Dynamic parallelism - prepare child kernel
    Ray *d_rays;
    HitInfo *d_hit_infos;
    cudaMalloc(&d_rays, width * height * sizeof(Ray));
    cudaMalloc(&d_hit_infos, width * height * sizeof(HitInfo));

    // note: we use serial device code for first hit as draft,
    // d_rays and d_hit_infos are not used in this kernel currently
    ray_trace_kernel<<<block_per_grid, thread_per_block>>>(*d_camera,
                                                           d_objects,
                                                           objects.size(),
                                                           d_lights,
                                                           lights.size(),
                                                           d_materials,
                                                           materials.size(),
                                                           width,
                                                           height,
                                                           d_rays,
                                                           d_hit_infos,
                                                           d_rgb_image);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error Ray Trace: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(h_rgb_image.data(), d_rgb_image, image_size, cudaMemcpyDeviceToHost);

    std::cout << "Writing image to rgb.ppm" << std::endl;
    write_ppm("rgb.ppm", h_rgb_image, width, height, 3);

    cudaFree(d_camera);
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_lights);
}
