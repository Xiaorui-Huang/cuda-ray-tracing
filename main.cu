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

#include "argp_util.h"
#include "read_json.h"
#include "write_ppm.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    // arguments and default see [[include/argp_util.h]] - commentLinks extension
    arguments args;

    argp_parse(&argp, argc, argv, 0, 0, &args);

    // host data
    Camera camera;
    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // Read a camera and scene description from given .json file
    readJson(args.filename, camera, objects, lights, materials);
    // usually it's 16:9 -> 1.77777778f
    unsigned int width = args.resolution * camera.width / camera.height;
    unsigned int height = args.resolution;

    //device data
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

    size_t scene_size = sizeof(Camera) + objects.size() * sizeof(Object) +
                        materials.size() * sizeof(Material) + lights.size() * sizeof(Light);

    dim3 block_per_grid(args.blocksize, args.blocksize);
    dim3 thread_per_block(ceil(width, block_per_grid.x), ceil(height, block_per_grid.y));
    // To ensure all pixels are processed, we round up the number of blocks (although it reduces occupancy - i.e. empty threads)

    // Dynamic parallelism - prepare child kernel (Doesn't work - no child and parent grid sync)
    // Malloc for child grid memory
    // Ray *d_rays; HitInfo *d_hit_infos; cudaMalloc(&d_rays, width * height * sizeof(Ray)); cudaMalloc(&d_hit_infos, width * height * sizeof(HitInfo));

    // // Vanilla Kernel launch
    // ray_trace_kernel<<<block_per_grid, thread_per_block>>>(*d_camera, d_objects, objects.size(), d_lights, lights.size(), d_materials, materials.size(), width, height, d_rgb_image);

    // clang-format off
    const int label_width = 15;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left << std::setw(label_width) << "Resolution:" << width << " x " << height << std::endl;
    std::cout << std::left << std::setw(label_width) << "Block size:" << args.blocksize << " x " << args.blocksize << std::endl;
    std::cout << std::left << std::setw(label_width) << "Grid size:" << thread_per_block.x << " x " << thread_per_block.y << std::endl;
    // clang-format on
    // use timed wrapped kernel launch
    float milliseconds = LaunchTimedKernel(ray_trace_kernel,
                                           block_per_grid,
                                           thread_per_block,
                                           0,
                                           0,
                                           *d_camera,
                                           d_objects,
                                           objects.size(),
                                           d_lights,
                                           lights.size(),
                                           d_materials,
                                           materials.size(),
                                           width,
                                           height,
                                           d_rgb_image);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error Ray Trace: " << cudaGetErrorString(err) << std::endl;
        std::cout << "Try to increase block size" << std::endl;
        exit(1);
    }
    // clang-format off
    std::cout << std::left << std::setw(label_width) << "Time:" << milliseconds << " ms" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Throughput:" << width * height / milliseconds / 1000 << " M rays/s" << std::endl;
    std::cout << std::left << std::setw(label_width) << "FPS:" << 1000 / milliseconds << " fps" << std::endl;
    std::cout << std::left << std::setw(label_width) << "Scene size:" << scene_size << " bytes" << std::endl;
    // clang-format on

    cudaMemcpy(h_rgb_image.data(), d_rgb_image, image_size, cudaMemcpyDeviceToHost);

    // write to ppm
    if (args.ppm) {
        std::cout << std::endl << "Writing image to rgb.ppm" << std::endl;
        write_ppm("rgb.ppm", h_rgb_image, width, height, 3);
    } else {
        // write to png
        std::cout << std::endl << "Writing image to rgb.png" << std::endl;
        stbi_write_png("rgb.png", width, height, 3, h_rgb_image.data(), width * 3);
    }

    cudaFree(d_camera);
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_lights);
    cudaFree(d_rgb_image);
}
