# Data Structures

```cpp
struct PlaneParams {
    Vec3 point;
    Vec3 normal;
    int materialIndex;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo, int objectIndex) const {
        // Plane-ray intersection algorithm
        // ... (intersection logic goes here)
        
        if (/* intersection detected */) {
            hitInfo.tNear = /* calculated tNear */;
            hitInfo.tFar = /* calculated tFar */;
            hitInfo.normal = normal; // Assuming the normal is normalized
            hitInfo.objectId = objectIndex;
            return true;
        }
        return false;
    }
};

struct SphereParams {
    Vec3 center;
    double radius;
    int materialIndex;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo, int objectIndex) const {
        // Sphere-ray intersection algorithm
        // ... (intersection logic goes here)
        
        if (/* intersection detected */) {
            hitInfo.tNear = /* calculated tNear */;
            // hitInfo.tFar is not typically used for spheres as we generally consider the first intersection point
            hitInfo.normal = /* calculate normal at intersection */;
            hitInfo.objectId = objectIndex;
            return true;
        }
        return false;
    }
};

struct TriangleParams {
    Vec3 corners[3];
    int materialIndex;
    int soupID;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo, int objectIndex) const {
        // Triangle-ray intersection algorithm
        // ... (intersection logic goes here)
        
        if (/* intersection detected */) {
            hitInfo.tNear = /* calculated tNear */;
            hitInfo.tFar = /* calculated tFar */;
            hitInfo.normal = /* calculate normal at intersection */;
            hitInfo.objectId = objectIndex;
            return true;
        }
        return false;
    }
};

struct TriangleSoupParams {
    int startOffset;
    int count;
    int materialIndex;

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo, int objectIndex, const TriangleParams* globalTriangles) const {
        // TriangleSoup-ray intersection algorithm
        // Iterate over the triangles in the soup and check each one
        // ... (intersection logic goes here)
        
        if (/* intersection detected */) {
            // Assuming intersection with the closest triangle is found
            hitInfo.tNear = /* calculated tNear */;
            hitInfo.tFar = /* calculated tFar */;
            hitInfo.normal = /* calculate normal at intersection */;
            hitInfo.objectId = objectIndex;
            return true;
        }
        return false;
    }
};

```

Now, we incorporate these into the `Object` structure, along with the `AABB`:

```cpp
struct AABB {
    Vec3 min; // Minimum point
    Vec3 max; // Maximum point

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo) const {
        // AABB-ray intersection algorithm
        float tNear = -FLT_MAX; // Start with the lowest possible value
        float tFar = FLT_MAX; // Start with the highest possible value

        // Test intersection with the 3 pairs of planes
        for (int i = 0; i < 3; ++i) {
            float invD = 1.0f / ray.direction[i];
            float t0 = (min[i] - ray.origin[i]) * invD;
            float t1 = (max[i] - ray.origin[i]) * invD;

            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            tNear = max(tNear, t0);
            tFar = min(tFar, t1);

            if (tNear > tFar || tFar < minT || maxT < tNear) {
                return false; // No valid intersection
            }
        }

        // At this point, if there is a valid intersection, we check if it's within the range
        if (tNear < maxT && tFar > minT) {
            hitInfo.tNear = max(tNear, minT);
            hitInfo.tFar = min(tFar, maxT);
            // The normal is not set here as it's not relevant for an AABB intersection check.
            // It should be set during the intersection with the actual object inside the AABB.
            return true;
        }

        return false; // No intersection within the specified range
    }
};


struct BVHNode {
    AABB box;
    int leftIndex;   // Index of the left child in the BVH node array
    int rightIndex;  // Index of the right child in the BVH node array
    int objectIndex; // Index of the associated object, -1 if it's not a leaf node
};

union ObjectData {
    PlaneParams plane;
    SphereParams sphere;
    TriangleParams triangle;
    TriangleSoupParams triangleSoup;
    // Constructors for each type could be added here for convenience
};

struct Object {
    ObjectData data;
    ObjectType type;
    AABB boundingBox;
    int objectIndex;  // Unique identifier for the object

    __device__ bool intersect(const Ray &ray, float minT, float maxT, HitInfo &hitInfo, const TriangleParams* globalTriangles = nullptr) const {
        if (!boundingBox.intersect(ray, minT, maxT, hitInfo)) {
            return false; // Bounding box check first for early exit
        }

        // Object-specific intersection
        switch (type) {
            case ObjectType::Plane:
                return data.plane.intersect(ray, minT, maxT, hitInfo, objectIndex);
            case ObjectType::Sphere:
                return data.sphere.intersect(ray, minT, maxT, hitInfo, objectIndex);
            case ObjectType::Triangle:
                return data.triangle.intersect(ray, minT, maxT, hitInfo, objectIndex);
            case ObjectType::TriangleSoup:
                return data.triangleSoup.intersect(ray, minT, maxT, hitInfo, objectIndex, globalTriangles);
            default:
                return false; // Unrecognized object type
        }
    }
};
```

In this updated structure:

- `intersect` in `Object` calls the corresponding intersect method based on the `ObjectType`.
- For `TriangleSoup`, we assume that there is a global array of `TriangleParams` that represents all triangles, which we need to pass into the `intersect` function to access the correct triangles in the soup.
- An `AABB` has been added to `Object` to represent the bounding volume of the geometry for use in acceleration structures like BVH.

Remember that this is a high-level view, and the actual intersection methods (`intersect`) would need to be implemented to perform the specific geometric intersection tests for each shape type against a ray. Also, memory management, especially when dealing with dynamic global memory for `TriangleParams`, needs careful attention.
