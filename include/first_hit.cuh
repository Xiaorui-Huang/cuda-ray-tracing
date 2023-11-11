#ifndef FIRST_HIT_CUH
#define FIRST_HIT_CUH

#include "Float3d.cuh"
#include "HitInfo.cuh"
#include "Object.cuh"
#include "Ray.h"

/**
 * @brief Find the first object that the ray intersects with. Uses reduction algorithm.
 * 
 * Should be called with `n/2` threads, where `n` is the number of objects.
 * i.e. 1D grid of 1D blocks. there grid size is halved
 * 
 * See reduce3 
 * https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
 * 
 * @param ray - `Input`
 * @param objects 
 * @param min_t 
 * @param max_t 
 * @param hit_info - `Output`
 * @param hit_info.object_id - `return` -1 if Ray doesn't intersects with any object or not, else object id
 */
__global__ void first_hit(const Ray &ray,
                          const Object *objects,
                          const size_t num_objects,
                          const float min_t,
                          const float max_t,
                          HitInfo &hit_info) {

    extern __shared__ HitInfo shared[];
    // HitInfo *shared_max = shared + blockDim.x;

    // we double the index to let one thread handle the first level reduction
    // int i = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    int i = threadIdx.x + blockIdx.x * (blockDim.x);
    int tid = threadIdx.x;

    // HitInfo intialization
    // float t;
    // float3d n;
    // int hit_id;

    float t = 1.0;
    float3d n = float3d();
    int hit_id = -1;

    // thread guard
    if (i < num_objects) {
        // create and assign data (This is where the algorithm is parallellized)

        // hit_id = objects[i].intersect(ray, min_t, max_t, t, n) ? i : -1;

        shared[tid] =
            (hit_id != -1) ? HitInfo(t, t, n, hit_id) : HitInfo(INFINITY, -INFINITY, float3d(), -1);

        // reduction to find the min_t base on t_near in HitInfo
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                // both are hits: pick the one with the smaller t_near
                if (shared[tid].object_id != -1 && shared[tid + stride].object_id != -1) {
                    shared[tid] = (shared[tid].t_near < shared[tid + stride].t_near)
                                      ? shared[tid]
                                      : shared[tid + stride];
                    shared[tid].t_far = max(shared[tid].t_far, shared[tid + stride].t_far);
                } else if (shared[tid + stride].object_id != -1) {
                    // [tid + stride] is a hit: pick it
                    shared[tid] = shared[tid + stride];
                } // else: if the [tid + stride] is miss or both are miss => do nothing
            }
        }

        if (tid == 0)
            hit_info = shared[0];
    }
}

#endif