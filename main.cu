#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "Camera.h"
#include "Light.cuh"
#include "Material.h"
#include "Object.cuh"

#include "Float3d.cuh"
#include "read_json.h"
#include "write_ppm.h"
#include "ray_trace.cuh"

#include <cuda_runtime.h>


__global__ void show_material(Material *d_materials) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Material: %f\n", d_materials[i].phong_exponent);
}

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
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
    // readJson(argc <= 1 ? "../data/inside-a-sphere.json" : argv[1], camera,
    // objects, lights, materials);

    Camera *d_camera;
    Object *d_objects;
    Material *d_materials;
    Light *d_lights;
    // unsigned char d_rgb_image[3 * width * height];

    // send data to GPU
    // very well suited for __constant__ use case, except the data is too large
    to_cuda(d_camera, &camera);
    to_cuda(d_objects, objects.data(), objects.size());
    to_cuda(d_materials, materials.data(), materials.size());
    to_cuda(d_lights, lights.data(), lights.size());

    std::cout << "Memory Cost: "
              << sizeof(Camera) + objects.size() * sizeof(Object) +
                     materials.size() * sizeof(Material) + lights.size() * sizeof(Light)
              << std::endl;

    // kernel test func
    show_material<<<1, materials.size()>>>(d_materials);


    std::vector<unsigned char> rgb_image(3 * width * height);

    // For each pixel (i,j)
    for (unsigned i = 0; i < height; ++i) {
        // std::cout << std::fixed << std::setprecision(2);
        // std::cout << "\rprogress: " << (double)i * 100 / height << "%" << std::flush;
        for (unsigned j = 0; j < width; ++j) {
            // Set background color
            float3d rgb(0, 0, 0);

            // Compute viewing ray
            // Ray ray;
            // generate_ray(camera, i, j, width, height, ray);

            // Shoot ray and collect color
            // raycolor(ray, 1.0, objects, lights, 0, rgb);

            // Write double precision color into image
            auto clamp = [](double s) -> double { return std::max(std::min(s, 1.0), 0.0); };
            rgb_image[0 + 3 * (j + width * i)] = 255.0 * clamp(rgb(0));
            rgb_image[1 + 3 * (j + width * i)] = 255.0 * clamp(rgb(1));
            rgb_image[2 + 3 * (j + width * i)] = 255.0 * clamp(rgb(2));
        }
    }
    std::cout << std::endl;

    write_ppm("rgb.ppm", rgb_image, width, height, 3);

    cudaFree(d_camera);
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_lights);
}
