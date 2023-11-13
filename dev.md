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

<https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/>

### Best Practices

<https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html>

### Dynamic Linker

<https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/>

<https://forums.developer.nvidia.com/t/the-cost-of-relocatable-device-code-rdc-true/47665/12>

<https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/>

### CUDA Feature Availability

<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability>

### Dynamic Parallelism

<https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/>

### Unified Memory

<https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>

## Design update

### Nov 11 2023

Since discoverying you cannot sync the child grid and parent grid, not only interms of memory, but also execution order. child is execute before parent exits. And only way within the kernel to have some control is using the cudaTail stream sync. but that is another kernel call by itself. 

cuda 11.6 deprecated on device cudaDevice sync...

the following it an attemp to split up kernels to allow synchronization. however, it gets complicated really fast. an better alternative is to have simple kernel with fast on device BVH traversal. which may prove to be faster considering the overhead of launching kernels and moving data between host and device.

a design like this will have a kernel per first hit call (for a pixel)
O(max_recursive_call * max_lights) number of kernels -> which is not horrible but very complicated

    // remodel recursion as stack
    // 0. first hit
    // 1. blind phong shading
    //      per light - first hit
    // 2. if recursive call < max recursive call
    // 3. push the reflected ray
    // 4. pop reflected ray and repeat
    

    // any ways... too complicated
    // hit_infos get over written every time we have kernel first hit, need to many memory copies 
    /**
     * while (stack is not empty) 
     *     pop ray
     *     kernel first hit <<<>>>
     *     hit_infos consolidate over every block, for all pixels (on device)
     *     bling phong ambient rgb<<<>>>
     *        per light - first hit kernel - add to rgb
     *        
     *      
     *        
     *     push reflected ray
     *
    */
