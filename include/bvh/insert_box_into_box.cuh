#ifndef INSERT_BOX_INTO_BOX_H
#define INSERT_BOX_INTO_BOX_H

#include "AABB.cuh"
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
#endif