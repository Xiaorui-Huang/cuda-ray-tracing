#ifndef UTIl_CUH
#define UTIl_CUH

#include <stdexcept>
#include <string>

/**
 * @brief Macro to launch a kernel and measure its execution time
 * 
 * @param kernelName Name of the kernel to launch 
 * @param gridDim Grid dimensions
 * @param blockDim Block dimensions
 * @param sharedMemBytes Size of shared memory in bytes
 * @param stream CUDA stream to use
 * @param ... Arguments to pass to the kernel
 * @return Execution time of the kernel in milliseconds (float)
 */
#define LaunchTimedKernel(kernelName, gridDim, blockDim, sharedMemBytes, stream, ...)              \
    ([&]() -> float {                                                                              \
        cudaEvent_t start, stop;                                                                   \
        float milliseconds = 0;                                                                    \
                                                                                                   \
        cudaEventCreate(&start);                                                                   \
        cudaEventCreate(&stop);                                                                    \
                                                                                                   \
        cudaEventRecord(start, stream);                                                            \
                                                                                                   \
        kernelName<<<gridDim, blockDim, sharedMemBytes, stream>>>(__VA_ARGS__);                    \
                                                                                                   \
        cudaEventRecord(stop, stream);                                                             \
                                                                                                   \
        cudaDeviceSynchronize();                                                                   \
                                                                                                   \
        cudaEventElapsedTime(&milliseconds, start, stop);                                          \
                                                                                                   \
        cudaEventDestroy(start);                                                                   \
        cudaEventDestroy(stop);                                                                    \
                                                                                                   \
        return milliseconds;                                                                       \
    })()

enum class Color { Red, Green, Blue };

/**
 * @brief Send Array of length `num` from Host to Device
 *
 * @tparam T Type of Array
 * @param d_ptr Pointer to Device Array
 * @param h_ptr Pointer to Host Array
 * @param num Length of Array
 */
template <typename T> inline void to_cuda(T *&d_ptr, const T *h_ptr, size_t num = 1) {
    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc(&d_ptr, sizeof(T) * num);
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA Error in cudaMalloc: " +
                                 std::string(cudaGetErrorString(err)));

    // Copy data from host to device
    err = cudaMemcpy(d_ptr, h_ptr, sizeof(T) * num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_ptr); // Clean up GPU memory allocation
        throw std::runtime_error("CUDA Error in cudaMemcpy: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <typename T> __device__ inline T min(T a, T b) { return a < b ? a : b; }

template <typename T> __device__ inline T max(T a, T b) { return a < b ? b : a; }

template <typename T> __host__ __device__ inline T ceil(T a, T b) { return (a + b - 1) / b; }

__host__ __device__ inline bool is_close(float a, float b, float tolerance = 1e-6f) {
    return fabsf(a - b) <= tolerance;
}

// debug only functions
//release set the NBDEBUG flag -> it is definedo
// debug does not set the NBDEBUG flag -> it is not defined
#ifdef DEBUG
__device__ inline float3d normal_to_rgb(const float3d &normal) { return normal * 0.5f + 0.5f; }
__device__ inline float3d depth_to_rgb(float t, float zNear, const float3d &ray_direction) {
    // Linearize the depth value based on the near plane and ray direction
    double linearized_depth = zNear / (t * ray_direction.norm());

    // Clamp the linearized depth to the range [0, 1]
    linearized_depth = linearized_depth < 1.0 ? linearized_depth : 1.0;

    // Convert the linearized depth to an RGB value
    return float3d(linearized_depth);
}
#endif

#endif