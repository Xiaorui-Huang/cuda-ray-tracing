#ifndef FIRST_HIT_CUH
#define FIRST_HIT_CUH

#include "Float3d.cuh"
#include "HitInfo.cuh"
#include "Object.cuh"
#include "Ray.h"

/**
 * @brief Find the first object that the ray intersects with. 
 * 
 * @param ray 
 * @param objects The array of objects in the scene 
 * @param num_objects Number of objects in the scene
 * @param min_t The minimum t value that the ray can have
 * @param max_t The maximum t value that the ray can have
 * @param hit_info `Output` - The hit info of the first object that the ray intersects with
 * @return true if ray intersects with any object, else false
 */
__device__ bool first_hit(const Ray &ray,
                          const Object *objects,
                          const size_t num_objects,
                          const float min_t,
                          const float max_t,
                          HitInfo &hit_info) {

    float smallest_t = INFINITY, furthest_t = -INFINITY;
    float cur_t;
    float3d cur_n;
    bool found = false;

    for (size_t i = 0; i < num_objects; i++) {
        if (objects[i].intersect(ray, min_t, max_t, cur_t, cur_n)) {
            found = true;
            if (cur_t < smallest_t) {
                smallest_t = cur_t;
                hit_info.normal = cur_n;
                hit_info.object_index = i;
            }
            if (cur_t > furthest_t)
                furthest_t = cur_t;
        }
    }

    if (found) {
        hit_info.t = smallest_t;
        hit_info.t_far = furthest_t;
    }

    return found;
}

/**
 * 😭🥲😥😢 Abandoned - since we can't reasonablly use dynamic parallelism without child and parent sync
 * @brief Kernel to find the first object that the ray intersects with. Uses reduction algorithm.
 * 
 * Should be called with `n/2` threads, where `n` is the number of objects.
 * i.e. 1D grid of 1D blocks. there grid size is halved
 * 
 * See reduce3 
 * https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction
 * 
 * @param ray 
 * @param objects 
 * @param min_t 
 * @param max_t 
 * @param hit_info - `Output`
 * @param hit_info.object_index - `return` -1 if Ray doesn't intersects with any object or not, else object id
 */
__global__ void first_hit(const Ray &ray,
                          const Object *objects,
                          const size_t num_objects,
                          const float min_t,
                          const float max_t,
                          HitInfo *hit_info) {

    extern __shared__ HitInfo shared[];
    // HitInfo *shared_max = shared + blockDim.x;

    // we double the index to let one thread handle the first level reduction
    // int i = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    int i = threadIdx.x + blockIdx.x * (blockDim.x);
    int tid = threadIdx.x;

    // HitInfo intialization
    float t;
    float3d n;
    int hit_id;

    // float t = 1.0;
    // float3d n = float3d();
    // int hit_id = -1;

    // thread guard
    if (i < num_objects) {
        // create and assign data (This is where the algorithm is parallellized)

        hit_id = objects[i].intersect(ray, min_t, max_t, t, n) ? i : -1;

        shared[tid] =
            (hit_id != -1) ? HitInfo(t, t, n, hit_id) : HitInfo(INFINITY, -INFINITY, float3d(), -1);

        // reduction to find the min_t base on t in HitInfo
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                // both are hits: pick the one with the smaller t
                if (shared[tid].object_index != -1 && shared[tid + stride].object_index != -1) {
                    shared[tid] = (shared[tid].t < shared[tid + stride].t)
                                      ? shared[tid]
                                      : shared[tid + stride];
                    shared[tid].t_far = max(shared[tid].t_far, shared[tid + stride].t_far);
                } else if (shared[tid + stride].object_index != -1) {
                    // [tid + stride] is a hit: pick it
                    shared[tid] = shared[tid + stride];
                } // else: if the [tid + stride] is miss or both are miss => do nothing
            }
        }

        // store the current block's result in global memory
        if (tid == 0)
            hit_info[blockIdx.x] = shared[0];
    }
}

#endif