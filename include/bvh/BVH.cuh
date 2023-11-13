#ifndef BVH_CUH
#define BVH_CUH

#include "AABB.cuh"

struct BVHNode {
    AABB box;
    int left_index;   // Index of the left child in the BVH node array
    int right_index;  // Index of the right child in the BVH node array
    int object_index; // Index of the associated object, -1 if it's not a leaf node

    __device__ BVHNode() : left_index(-1), right_index(-1), object_index(-1) {}

    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const {}
};

// return the bvh root node index
/*
int constructBVH(std::vector<BVHNode> &bvhNodes,
                 const std::vector<Object> &objects,
                 int start,
                 int end) {
    BVHNode node;
    bvhNodes.reserve(objects.size() * 2); // a perfect tree would have 2n-1 nodes
    int nodeIndex = bvhNodes.size();

    bvhNodes.push_back(node); // Placeholder for the current node

    // Compute the bounding box for the current set of objects

    for (const auto &obj : objects) {
        box.min= box.min_corner.cwiseMin(obj->box.min_corner);
        box.max= box.max_corner.cwiseMax(obj->box.max_corner);
    }

    // for (int i = start; i < end; ++i) {
    //     bvhNodes[nodeIndex].box = insert_box_into_box(bvhNodes[nodeIndex].box, objects[i].box);
    // }

    int numObjects = end - start;
    if (numObjects == 1) {
        // Leaf node
        bvhNodes[nodeIndex].object_index = start;
        return nodeIndex;
    }

    // Find the midpoint of the bounding box
    float3d midPoint = bvhNodes[nodeIndex].box.center();

    // Partition the objects around the midpoint
    int mid = partitionObjects(objects, start, end, midPoint);

    // Recursively construct child nodes
    bvhNodes[nodeIndex].left_index = constructBVH(bvhNodes, objects, start, mid);
    bvhNodes[nodeIndex].right_index = constructBVH(bvhNodes, objects, mid, end);

    return nodeIndex;
}

// Function to partition objects based on a midpoint
int partitionObjects(const std::vector<Object> &objects,
                     int start,
                     int end,
                     const float3d &midPoint) {
    // Implement partition logic based on object positions and the midpoint
    // ...
}

// Function to compute the union of two AABBs
void insert_box_into_box(const AABB &A, AABB &B) {
    // checking for empty bounds along all axis
    // where empty bounds are defined as: tex: max - min < 0

    // TODO: investigate might be redundant
    if (((B.max - B.min) < 0.0).any()) {
        B.min = A.min;
        B.max = A.max;
    }
    B.min = B.min.cwiseMin(A.min);
    B.max = B.max.cwiseMax(A.max);
}
*/

#endif
