# Development Notes

## 1. Cpp to Cuda Data Transfer

CUDA as a parallel programming paradigm, works best with simple inline, predictable datastructures.
Therefore dynamic array to pointers to virtual base classes that points to concrete child class with
dynamic sizes will not work.

Significant Effort when into redesigning the necessary parts that interfaces with defining a rich scene.

## 2. Bounding Volume Hierarchy

Two Blog that give significant insight into the implementation of parallel BVH


[BVH Traversal](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)

[BVH Construction](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)

## 3. Shading 

## 4. Profiling


## Misc

<https://developer.nvidia.com/blog/cooperative-groups/>

<https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/>

<https://developer.nvidia.com/blog/recreate-high-fidelity-digital-twins-with-neural-kernel-surface-reconstruction/>

<https://developer.nvidia.com/blog/multi-gpu-programming-with-standard-parallel-c-part-1/>

<https://www.nvidia.com/en-us/events/cvpr/>

<http://www.kevinbeason.com/smallpt/>

<https://www.pbrt.org/>