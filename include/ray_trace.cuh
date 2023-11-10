#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH

#include "viewing_ray.cuh"

// Kernel that performs ray tracing
__global__ void rayTraceKernel(const Camera &camera, const size_t width, const size_t height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index, left to right
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index, top to bottom
    if (col < width && row < height) {

        Ray ray;
        generateRay(camera, row, col, width, height, ray);
    }
}

#endif