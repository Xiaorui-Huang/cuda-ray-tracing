#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() { printf("Hello World from GPU!\n"); }

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
