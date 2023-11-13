#ifndef BVH_CUH
#define BVH_CUH

#include "AABB.cuh"
#include "Object.cuh"
#include <cassert>
#include <memory>
#include <vector>

struct ObjectState {
    Object obj;
    size_t index;
    
    ObjectState(const Object &obj, size_t index) : obj(obj), index(index) {}
};
struct BVHNode {
    AABB box;
    int left_index;  // Index of the left child in the BVH node array, -1 if there is no left child
    int right_index; // Index of the right child in the BVH node array, -1 if there is no right child
    int object_index; // Index of the associated object, -1 if it's not a leaf node

    // Case 1: leaf node,       object_index != -1, left_index == -1, right_index == -1
    // Case 2: internal node,   object_index == -1, left_index != -1, right_index != -1

    __host__ __device__ BVHNode() : left_index(-1), right_index(-1), object_index(-1) {}

    __device__ bool
    intersect(const Ray &ray, const float min_t, const float max_t, float &t, float3d &n) const {
        
    }
};

// insert box A into box B - expand box B to fit box A
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

// Naive CPU Construction - will not sort the objects
// return the bvh root node index and use populate the bvh_nodes vector
int constructBVH_recurse(const std::vector<std::shared_ptr<ObjectState>> &states,
                         std::vector<BVHNode> &bvh_nodes,
                         size_t &bvh_size) {
    BVHNode node;
    auto num_leaves = states.size();

    int node_index = bvh_size++;

    // leaf node - a concrete object
    if (num_leaves == 1) {
        node.object_index = states[0]->index;
        insert_box_into_box(states[0]->obj.box, node.box);
        bvh_nodes.at(node_index) = node;
        return node_index;
    }

    // Compute the bounding box for the current set of objects
    for (const auto &state : states) {
        node.box.min = node.box.min.cwiseMin(state->obj.box.min);
        node.box.max = node.box.max.cwiseMax(state->obj.box.max);
    }

    int axis = -1;
    float cur_max = -1;
    auto span = node.box.max - node.box.min;

    // find the longest axis
    for (int i = 0; i < 3; i++) {
        if (span[i] > cur_max) {
            cur_max = span[i];
            axis = i;
        }
    }

    float mid = (node.box.max[axis] + node.box.min[axis]) * 0.5;

    // partition the objects
    std::vector<std::shared_ptr<ObjectState>> left_list, right_list, center_list;

    for (const auto &shared : states) {
        if (shared->obj.box.center()[axis] < mid)
            left_list.push_back(shared);
        else if (shared->obj.box.center()[axis] > mid)
            right_list.push_back(shared);
        else
            center_list.push_back(shared);
    }

    // Handle edge cases and reassign left and right lists accordingly
    if (center_list.size() == num_leaves) {
        // All centers are mid-point; split list into two
        auto center_iter = center_list.begin() + center_list.size() / 2;
        left_list.assign(center_list.begin(), center_iter);
        right_list.assign(center_iter, center_list.end());
    } else if (left_list.empty() || right_list.empty()) {
        // Either left or right is empty; reassign center list
        auto &target_list = left_list.empty() ? left_list : right_list;
        target_list.insert(target_list.end(), center_list.begin(), center_list.end());
    } else {
        // Neither left nor right list is empty
        auto &target_list = (left_list.size() < right_list.size()) ? left_list : right_list;
        target_list.insert(target_list.end(), center_list.begin(), center_list.end());
    }

    assert(left_list.size() + right_list.size() == num_leaves);
    assert(!left_list.empty());
    assert(!right_list.empty());

    // Find the midpoint of the bounding box
    node.left_index = constructBVH_recurse(left_list, bvh_nodes, bvh_size);
    node.right_index = constructBVH_recurse(right_list, bvh_nodes, bvh_size);

    insert_box_into_box(bvh_nodes[node.left_index].box, node.box);
    insert_box_into_box(bvh_nodes[node.right_index].box, node.box);

    // after all the values are populated, set the node to the correct values
    bvh_nodes.at(node_index) = node;

    return node_index;
}

int constructBVH(const std::vector<Object> &objects, std::vector<BVHNode> &bvh_nodes) {
    std::vector<std::shared_ptr<ObjectState>> object_states;
    object_states.reserve(objects.size());
    // number of nodes in the bvh tree
    // calculated total number of nodes in the tree
    bvh_nodes.resize(objects.size() * 2 - 1); // a perfect tree would have 2n-1 nodes

    size_t bvh_size = 0;

    // using this avoids copyting concrete objects over and over in the recursion
    // only pointers are copied
    for (size_t i = 0; i < objects.size(); i++)
        object_states.push_back(std::make_shared<ObjectState>(objects[i], i));

    int root_index = constructBVH_recurse(object_states, bvh_nodes, bvh_size);
    assert(bvh_size == bvh_nodes.size());
    return root_index;
}

#endif