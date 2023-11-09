#ifndef UTIl_H
#define UTIl_H


template <typename T>
__device__ inline T min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ inline T max(T a, T b) { return a < b ? b : a; }

#endif