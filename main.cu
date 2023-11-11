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
    int width = 640;
    int height = 360;

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
    to_cuda(d_materials, materials.data(), materials.size());
    to_cuda(d_lights, lights.data(), lights.size());
    // Good use for 2D or 3D array, but accessing is too complicated
    // https: //docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cudaMallocPitched#device-memory
    cudaMalloc(&d_rgb_image, image_size);

    std::cout << "Memory Cost: "
              << sizeof(Camera) + objects.size() * sizeof(Object) +
                     materials.size() * sizeof(Material) + lights.size() * sizeof(Light)
              << std::endl;

    // kernel test func
    show_material<<<1, materials.size()>>>(d_materials);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error Show Material: " << cudaGetErrorString(err) << std::endl;

    dim3 block_per_grid(N, N);
    // To ensure all pixels are processed, we round up the number of blocks (reduces occupancy - i.e. empty threads)
    // Note: int/int = int, so we need to cast to float before division
    dim3 thread_per_block(std::ceil((float)width / block_per_grid.x),
                          std::ceil((float)height / block_per_grid.y));

    // dim3 thread_per_block((width + block_per_grid.x - 1) / block_per_grid.x,
    //                       (height + block_per_grid.y - 1) / block_per_grid.y);

    ray_trace_kernel<<<block_per_grid, thread_per_block>>>(
        *d_camera, d_objects, objects.size(), d_lights, lights.size(), width, height, d_rgb_image);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error Ray Trace: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(h_rgb_image.data(), d_rgb_image, image_size, cudaMemcpyDeviceToHost);

    // // For each pixel (i,j)
    // for (unsigned i = 0; i < height; ++i) {
    //     // std::cout << std::fixed << std::setprecision(2);
    //     // std::cout << "\rprogress: " << (double)i * 100 / height << "%" << std::flush;
    //     for (unsigned j = 0; j < width; ++j) {
    //         // Set background color
    //         float3d rgb(0, 0, 0);

    //         // Compute viewing ray
    //         // Ray ray;
    //         // generate_ray(camera, i, j, width, height, ray);

    //         // Shoot ray and collect color
    //         // raycolor(ray, 1.0, objects, lights, 0, rgb);

    //         // Write double precision color into image
    //         auto clamp = [](double s) -> double { return std::max(std::min(s, 1.0), 0.0); };
    //         h_rgb_image[0 + 3 * (j + width * i)] = 255.0 * clamp(rgb(0));
    //         h_rgb_image[1 + 3 * (j + width * i)] = 255.0 * clamp(rgb(1));
    //         h_rgb_image[2 + 3 * (j + width * i)] = 255.0 * clamp(rgb(2));
    //     }
    // }
    // std::cout << std::endl;

    std::cout << "Writing image to rgb.ppm" << std::endl;
    write_ppm("rgb.ppm", h_rgb_image, width, height, 3);

    cudaFree(d_camera);
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_lights);
}
