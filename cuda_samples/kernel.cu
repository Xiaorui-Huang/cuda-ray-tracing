#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N = 3;
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    h_A[0] = 1.0;
    h_A[1] = 2.0;
    h_A[2] = 3.0;
    h_B[0] = 4.0;
    h_B[1] = 5.0;
    h_B[2] = 122.0;

    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vec_add<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    std::cout << h_C[0] << " " << h_C[1] << " " << h_C[2] << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}