To design and store objects for ray tracing in CUDA, you'll want to follow a pattern that minimizes the drawbacks of polymorphism on the device. Here's a general approach to the problem:

1. **Object Representation**: Define a struct for each object type that stores the specific parameters needed to represent it (e.g., center and radius for spheres, point and normal for planes).

2. **Union of Parameters**: Create a union that can hold the parameters for any of the object types. This way, you have a consistent memory layout that can be efficiently accessed in CUDA.

3. **Type Enumeration**: Use an enumeration to keep track of the object type. This way, you can use a switch statement in the kernel to differentiate between object types and execute the appropriate intersection code.

4. **Flattened Data Structure**: You'll want to flatten your object hierarchy into a data structure that can be easily copied to the CUDA device.

5. **Intersection Function**: Write a device function to handle the intersection test for each object type. The function will use the object's type to determine which intersection logic to execute.

Here is a rough outline of how this might look in code:

```cpp
enum class ObjectType {
    PLANE,
    SPHERE,
    TRIANGLE,
    TRIANGLE_SOUP // Assuming triangle soup is an array of triangles
};

struct PlaneParams {
    Vec3d point;
    Vec3d normal;
};

struct SphereParams {
    Vec3d center;
    double radius;
};

struct TriangleParams {
    Vec3d vertices[3];
};

struct TriangleSoupParams {
    TriangleParams* triangles;
    int numTriangles;
};

union ObjectData {
    PlaneParams plane;
    SphereParams sphere;
    TriangleParams triangle;
    TriangleSoupParams triangleSoup;
};

struct Object {
    ObjectType type;
    ObjectData data;
};

__device__ bool intersect(const Ray &ray, const Object &obj, double &t, Vec3d &n) {
    switch (obj.type) {
        case ObjectType::PLANE:
            return intersectPlane(ray, obj.data.plane, t, n);
        case ObjectType::SPHERE:
            return intersectSphere(ray, obj.data.sphere, t, n);
        // ... other cases for triangles, etc.
    }
    return false;
}

// Kernel to perform ray tracing
__global__ void rayTraceKernel(Object* objects, int numObjects, /* ... other params ... */) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;

    Ray ray = /* ... compute ray based on thread index ... */;
    double t;
    Vec3d n;
    if (intersect(ray, objects[idx], t, n)) {
        // Handle intersection
    }
}

// Host function to copy objects to the device
Object* copyObjectsToDevice(const std::vector<std::shared_ptr<Object>>& hostObjects) {
    Object* deviceObjects;
    Object* hostBuffer = new Object[hostObjects.size()];

    // Populate the host buffer with Object structs
    for (size_t i = 0; i < hostObjects.size(); ++i) {
        hostBuffer[i].type = /* ... determine type from hostObjects[i] ... */;
        hostBuffer[i].data = /* ... populate data based on type ... */;
    }

    cudaMalloc(&deviceObjects, hostObjects.size() * sizeof(Object));
    cudaMemcpy(deviceObjects, hostBuffer, hostObjects.size() * sizeof(Object), cudaMemcpyHostToDevice);

    delete[] hostBuffer;
    return deviceObjects;
}
```

This design eliminates the need for virtual functions and polymorphism on the CUDA device, which could otherwise be a source of significant overhead. You will need to translate the logic specific to each shape into CUDA-compatible code and make sure that the union `ObjectData` is properly populated based on the object type. 

This is a simplified example. You'll need to adapt it to handle materials, transformations, and other properties of your objects. You might also need to manage the dynamic memory for objects like `TriangleSoup` carefully, ensuring you allocate and free memory on the device as needed.