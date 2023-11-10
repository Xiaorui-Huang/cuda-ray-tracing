#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "Camera.h"
#include "Vec3d.cuh"
// #include "Light.h"
#include "Object.cuh"
// #include "raycolor.h"
#include "read_json.h"
// #include "viewing_ray.h"
#include "write_ppm.h"

int main(int argc, char *argv[]) {
    Camera camera;
    std::vector<Object> objects;
    std::vector<Material> materials;

    // Read a camera and scene description from given .json file

    read_json(argc <= 1 ? "../data/bunny.json" : argv[1], camera, objects, materials);
    // read_json(argc <= 1 ? "../data/inside-a-sphere.json" : argv[1], camera, objectsVec,
    // lightsVec);

    Object *dObjects = nullptr; // Pointer to device memory.

    // Allocate memory on the device.
    cudaMalloc(&dObjects, objects.size() * sizeof(Object));

    // Copy data from host to device.
    cudaMemcpy(dObjects, objects.data(), objects.size() * sizeof(Object),
               cudaMemcpyHostToDevice);

    // Now dObjects can be used in CUDA kernels.

    // ... (Perform computations with dObjects in CUDA kernels)

    // If you need to copy the results back to the host:
    cudaMemcpy(objects.data(), dObjects, objects.size() * sizeof(Object),
               cudaMemcpyDeviceToHost);

    // Free the memory on the device when you're done with it.
    cudaFree(dObjects);

    int width = 640;
    int height = 360;
    std::vector<unsigned char> rgb_image(3 * width * height);

    // For each pixel (i,j)
    for (unsigned i = 0; i < height; ++i) {
        // std::cout << std::fixed << std::setprecision(2);
        // std::cout << "\rprogress: " << (double)i * 100 / height << "%" << std::flush;
        for (unsigned j = 0; j < width; ++j) {
            // Set background color
            Vec3d rgb(0, 0, 0);

            // Compute viewing ray
            // Ray ray;
            // viewing_ray(camera, i, j, width, height, ray);

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
}
