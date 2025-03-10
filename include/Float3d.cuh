#ifndef FLOAT3D_CUH
#define FLOAT3D_CUH

#include <cmath>

__host__ __device__ inline bool is_zero(float a, float tolerance = 1e-6f) {
    return fabsf(a) <= tolerance;
}

struct __align__(16) float3d {
    union {
        struct {
            float x, y, z;
        };
        float data[3];
    };
    float padding; // for 16 byte alignment, for Coalesced memory access

    __host__ __device__ float3d() : data{0.0, 0.0, 0.0} {}

    __host__ __device__ float3d(float x) : data{x, x, x} {}

    __host__ __device__ float3d(float x, float y, float z) : data{x, y, z} {}

    // __host__ __device__ inline float &x() { return data[0]; }
    // __host__ __device__ inline const float &x() const { return data[0]; }

    // __host__ __device__ inline float &y() { return data[1]; }
    // __host__ __device__ inline const float &y() const { return data[1]; }

    // __host__ __device__ inline float &z() { return data[2]; }
    // __host__ __device__ inline const float &z() const { return data[2]; }

    __host__ __device__ inline float &operator[](size_t index) { return data[index]; }

    __host__ __device__ inline const float &operator[](size_t index) const { return data[index]; }

    __host__ __device__ inline float &operator()(size_t index) { return data[index]; }

    __host__ __device__ inline const float &operator()(size_t index) const { return data[index]; }

    // Addition
    __host__ __device__ inline float3d operator+(const float3d &other) const {
        return float3d(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ inline float3d &operator+=(const float3d &other) {
        data[0] += other.x;
        data[1] += other.y;
        data[2] += other.z;
        return *this;
    }

    // Subtraction
    __host__ __device__ inline float3d operator-(const float3d &other) const {
        return float3d(x - other.x, y - other.y, z - other.z);
    }

    /// Unary minus
    __host__ __device__ inline float3d operator-() const {
        return float3d(-data[0], -data[1], -data[2]);
    }

    // Scalar multiplication
    __host__ __device__ inline float3d operator*(float scalar) const {
        return float3d(x * scalar, y * scalar, z * scalar);
    }

    // Element-wise multiplication
    __host__ __device__ inline float3d operator*(const float3d &other) const {
        return float3d(x * other.x, y * other.y, z * other.z);
    }

    // Scalar division
    __host__ __device__ inline float3d operator/(float scalar) const {
        float divisor = 1.0 / scalar;
        return (*this) * divisor;
    }

    // Dot product
    __host__ __device__ inline float dot(const float3d &other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    __host__ __device__ inline float3d cross(const float3d &other) const {
        return float3d(y * other.z - z * other.y,
                       z * other.x - x * other.z,
                       x * other.y - y * other.x);
    }

    __host__ __device__ inline float3d operator<(const float3d &other) const {
        return float3d(x < other.x, y < other.y, z < other.z);
    }

    __host__ __device__ inline float3d operator>(const float3d &other) const {
        return float3d(x > other.x, y > other.y, z > other.z);
    }

    __host__ __device__ inline float3d operator<=(const float3d &other) const {
        return float3d(x <= other.x, y <= other.y, z <= other.z);
    }

    __host__ __device__ inline float3d operator>=(const float3d &other) const {
        return float3d(x >= other.x, y >= other.y, z >= other.z);
    }

    __host__ __device__ inline bool any() const {
        return !is_zero(x) || !is_zero(y) || !is_zero(z);
    }

    __host__ __device__ inline float norm() const { return sqrt(dot(*this)); }

    __host__ __device__ inline float3d normalized() const {
        float length2 = dot(*this);
        if (is_zero(length2)) return float3d(0.0f);  // Avoid division by zero
        return (*this) * rsqrtf(length2);  // Faster than `1.0f / sqrtf(length2)`
    }

    __host__ __device__ inline void normalize() {
        float length2 = dot(*this);
        if (!is_zero(length2)) 
            (*this)= (*this) * rsqrtf(length2);
    }

    __host__ __device__ inline float3d cwiseMax(const float3d &other) const {
        return float3d(fmaxf(x, other.x), fmaxf(y, other.y), fmaxf(z, other.z));
    }
    
    __host__ __device__ inline float3d cwiseMin(const float3d &other) const {
        return float3d(fminf(x, other.x), fminf(y, other.y), fminf(z, other.z));
    }
};

__host__ __device__ inline float3d operator*(float scalar, const float3d &vec) {
    return vec * scalar;
}

#endif // float3d_CUH