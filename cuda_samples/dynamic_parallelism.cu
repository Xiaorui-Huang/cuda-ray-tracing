#include <cuda_runtime.h>
#include <iostream>

// Kernel definitions
__global__ void tail_launch(int *data) { data[threadIdx.x] = data[threadIdx.x] + 1; }

__global__ void child_launch(int *data) { data[threadIdx.x] = data[threadIdx.x] + 1; }

__global__ void parent_launch(int *data) {
    data[threadIdx.x] = threadIdx.x;

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Before child_launch - data[203]: %d\n", data[203]);

        child_launch<<<1, 256>>>(data);
        printf("child_launch - data[203]: %d\n", data[203]);
        // child_launch<<<1, 256>>>(data);
        // printf("child_launch - data[203]: %d\n", data[203]);
        // child_launch<<<1, 256>>>(data);
        // printf("child_launch - data[203]: %d\n", data[203]);

        // // these are how you would do it in the host code, Not applicable
        // cudaStream_t stream;
        // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        tail_launch<<<1, 256, 0, cudaStreamTailLaunch>>>(data);
        printf("child_launch - data[203]: %d\n", data[203]);
        tail_launch<<<1, 256, 0, cudaStreamTailLaunch>>>(data);
        printf("child_launch - data[203]: %d\n", data[203]);
        tail_launch<<<1, 256, 0, cudaStreamTailLaunch>>>(data);
        printf("child_launch - data[203]: %d\n", data[203]);

        // cudaStreamSynchronize(stream);
        // cudaStreamDestroy(stream);
    }
}

// Host function to launch the parent kernel

// Main function
int main() {
    int *data;
    int N = 256;
    size_t size = N * sizeof(int);

    // Allocate memory on the GPU
    cudaMalloc(&data, size);

    // Launch the parent kernel from the host
    parent_launch<<<1, 256>>>(data);

    // Allocate memory on the host for the result
    int *result = new int[N];

    // Copy the result back to the host
    cudaMemcpy(result, data, size, cudaMemcpyDeviceToHost);

    // Print the result
    // for (int i = 0; i < N; ++i) {
        // std::cout << "Data[" << i << "] = " << result[i] << std::endl;
    // }
    printf("Data[203] = %d\n", result[203]);

    // Clean up
    delete[] result;
    cudaFree(data);

    return 0;
}
