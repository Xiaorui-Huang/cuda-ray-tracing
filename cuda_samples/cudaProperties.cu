#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    int device;

    // Get the device id of the currently active GPU.
    cudaGetDevice(&device);

    // Get the properties of the device.
    cudaGetDeviceProperties(&prop, device);

    // Print out the size of the constant and shared memory.
    std::cout << "Total constant memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    return 0;
}
