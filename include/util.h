#ifndef UTIl_H
#define UTIl_H

/**
 * @brief Send Array of length `num` from Host to Device
 *
 * @tparam T Type of Array
 * @param d_Ptr Pointer to Device Array
 * @param h_Ptr Pointer to Host Array
 * @param num
 */
template <typename T> inline void toCuda(T *&d_Ptr, T *h_Ptr, size_t num = 1) {
    cudaMalloc(&d_Ptr, sizeof(T) * num);
    cudaMemcpy(d_Ptr, h_Ptr, sizeof(T) * num, cudaMemcpyHostToDevice);
}

template <typename T>
__device__ inline T min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ inline T max(T a, T b) { return a < b ? b : a; }

#endif