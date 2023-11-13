#ifndef BVH_CUH
#define BVH_CUH

#include "AABB.cuh"

struct BVHNode {
    AABB box;
    int left_index;   // Index of the left child in the BVH node array
    int right_index;  // Index of the right child in the BVH node array
    int object_index; // Index of the associated object, -1 if it's not a leaf node
    
    __device__ BVHNode() : left_index(-1), right_index(-1), object_index(-1) {}
    
    __device__ bool intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const{

    }
};



#endif
