#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "Float3d.cuh"
#include "Light.cuh"
#include "Object.cuh"
#include <bvh/BVH.cuh>

#include "Camera.h"
#include "Material.h"

#include "ray_trace.cuh"

#include "argp_util.h"
#include "read_json.h"
#include "write_ppm.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

#define LABEL_WIDTH 15

int main(int argc, char *argv[]) {
    // arguments and default see [[include/argp_util.h]] - commentLinks extension
    arguments args;

    argp_parse(&argp, argc, argv, 0, 0, &args);

    // host data
    Camera camera;
    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // filter out planes for BVH
    std::vector<Plane> planes;

    // Read a camera and scene description from given .json file
    readJson(args.filename, camera, objects, lights, materials, planes, args.no_bvh);

    if (!args.no_bvh) {
        std::vector<BVHNode> bvh_nodes;
        size_t root_index = constructBVH(objects, bvh_nodes);

        auto box = bvh_nodes[root_index].box;
        auto center = bvh_nodes[root_index].box.center();
        // clang-format off
        std::cout << "[debug info - single .stl]:" << std::endl;
        std::cout << std::left << std::setw(LABEL_WIDTH) << "BVH root box: (" << box.min.x << ", " << box.min.y << ", " << box.min.z << ") - (" << box.max.x << ", " << box.max.y << ", " << box.max.z << ")" << std::endl;
        std::cout << std::left << std::setw(LABEL_WIDTH) << "Center of BVH root: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl << std::endl;
        // clang-format on
    }

    // usually it's 16:9 -> 1.77777778f
    unsigned int width = args.resolution * camera.width / camera.height;
    unsigned int height = args.resolution;
    if (args.landscape) {
        std::swap(width, height);
        std::swap(camera.width, camera.height);
    }

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

    // set by block or grid size
    // To ensure all pixels are processed, we round up the number of blocks (although it reduces occupancy - i.e. empty threads)
    dim3 block_dim = args.size;
    dim3 grid_dim(ceil_div(width, block_dim.x), ceil_div(height, block_dim.y));
    if (args.gridsize_set)
        std::swap(grid_dim, block_dim);

    // Dynamic parallelism - prepare child kernel (Doesn't work - no child and parent grid sync)
    // Malloc for child grid memory
    // Ray *d_rays; HitInfo *d_hit_infos; cudaMalloc(&d_rays, width * height * sizeof(Ray)); cudaMalloc(&d_hit_infos, width * height * sizeof(HitInfo));

    // // Vanilla Kernel launch
    // ray_trace_kernel<<<grid_dim, block_dim>>>(*d_camera, d_objects, objects.size(), d_lights, lights.size(), d_materials, materials.size(), width, height, d_rgb_image);

    // clang-format off
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Resolution:" << width << " x " << height << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Grid size:" << grid_dim.x << " x " << grid_dim.y << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Block size:" << block_dim.x << " x " << block_dim.y << std::endl;
    // clang-format on
    // use timed wrapped kernel launch
    float ms = TIME_KERNEL(0, {
        ray_trace_kernel<<<grid_dim, block_dim>>>(*d_camera,
                                                  d_objects,
                                                  objects.size(),
                                                  d_lights,
                                                  lights.size(),
                                                  d_materials,
                                                  materials.size(),
                                                  width,
                                                  height,
                                                  d_rgb_image);
    });

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error Ray Trace: " << cudaGetErrorString(err) << std::endl;
        std::cout << "Try to increase block size" << std::endl;
        exit(1);
    }
    // clang-format off
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Time:" << ms << " ms (" << ms/1000 << " s)" << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Throughput:" << width * height / ms / 1000 << " M rays/s" << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "FPS:" << 1000 / ms << " fps" << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "Scene size:" << scene_size << " bytes" << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "# of objects:" << objects.size() << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "# of lights:" << lights.size() << std::endl;
    std::cout << std::left << std::setw(LABEL_WIDTH) << "BVH enabled:" << (args.no_bvh ? "False" : "True") << std::endl;
    // clang-format on

    cudaMemcpy(h_rgb_image.data(), d_rgb_image, image_size, cudaMemcpyDeviceToHost);

    std::cout << std::endl << "Writing image to " << args.outputname << std::endl;

    if (ends_with(args.outputname, ".png"))
        stbi_write_png(args.outputname, width, height, 3, h_rgb_image.data(), width * 3);
    else if (ends_with(args.outputname, ".ppm"))
        write_ppm(args.outputname, h_rgb_image, width, height, 3);

    cudaFree(d_camera);
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_lights);
    cudaFree(d_rgb_image);
}
