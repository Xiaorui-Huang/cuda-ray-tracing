#include <cuda_runtime.h>
#include <iostream>

#define N 256 // example matrix size

// Kernel to add two matrices
__global__ void MatAdd(float MatA[N][N], float MatB[N][N], float MatC[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        MatC[i][j] = MatA[i][j] + MatB[i][j];
}

int main() {
    float MatA[N][N], MatB[N][N], MatC[N][N];

    // Initializing matrices with some values for testing
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            MatA[i][j] = i + j;
            MatB[i][j] = i - j;
        }

    float(*d_MatA)[N], (*d_MatB)[N], (*d_MatC)[N];

    // Allocate memory on the device
    cudaMalloc((void **)&d_MatA, N * N * sizeof(float));
    cudaMalloc((void **)&d_MatB, N * N * sizeof(float));
    cudaMalloc((void **)&d_MatC, N * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_MatA, MatA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, MatB, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_MatA, d_MatB, d_MatC);

    // Copy result matrix from device to host
    cudaMemcpy(MatC, d_MatC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // (Optional) Print the result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << MatC[i][j] << " ";
        std::cout << std::endl;
    }

    // Cleanup and free memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    return 0;
}
