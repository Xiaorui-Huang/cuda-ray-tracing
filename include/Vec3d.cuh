#ifndef VEC3D_CUH
#define VEC3D_CUH

#include <cmath> 

class Vec3d {
  private:
    float data[3];

  public:
    __host__ __device__ Vec3d() : data{0.0, 0.0, 0.0} {}

    __host__ __device__ Vec3d(float x) : data{x, x, x} {}

    __host__ __device__ Vec3d(float x, float y, float z) : data{x, y, z} {}

    __host__ __device__ inline float &x() { return data[0]; }
    __host__ __device__ inline const float &x() const { return data[0]; }

    __host__ __device__ inline float &y() { return data[1]; }
    __host__ __device__ inline const float &y() const { return data[1]; }

    __host__ __device__ inline float &z() { return data[2]; }
    __host__ __device__ inline const float &z() const { return data[2]; }

    __host__ __device__ inline float &operator[](size_t index) { return data[index]; }

    __host__ __device__ inline const float &operator[](size_t index) const { return data[index]; }

    __host__ __device__ inline float &operator()(size_t index) { return data[index]; }

    __host__ __device__ inline const float &operator()(size_t index) const { return data[index]; }

    // Addition
    __host__ __device__ inline Vec3d operator+(const Vec3d &other) const {
        return Vec3d(x() + other.x(), y() + other.y(), z() + other.z());
    }

    // Subtraction
    __host__ __device__ inline Vec3d operator-(const Vec3d &other) const {
        return Vec3d(x() - other.x(), y() - other.y(), z() - other.z());
    }

    /// Unary minus
    __host__ __device__ inline Vec3d operator-() const { 
        return Vec3d(-data[0], -data[1], -data[2]); 
    }

    // Scalar multiplication
    __host__ __device__ inline Vec3d operator*(float scalar) const {
        return Vec3d(x() * scalar, y() * scalar, z() * scalar);
    }


    // Element-wise multiplication
    __host__ __device__ inline Vec3d operator*(const Vec3d &other) const {
        return Vec3d(x() * other.x(), y() * other.y(), z() * other.z());
    }

    // Scalar division
    __host__ __device__ inline Vec3d operator/(float scalar) const {
        float devisor = 1.0 / scalar;
        return (*this) * devisor;
    }

    // Dot product
    __host__ __device__ inline float dot(const Vec3d &other) const {
        return x() * other.x() + y() * other.y() + z() * other.z();
    }

    // Cross product
    __host__ __device__ inline Vec3d cross(const Vec3d &other) const {
        return Vec3d(y() * other.z() - z() * other.y(), z() * other.x() - x() * other.z(),
                     x() * other.y() - y() * other.x());
    }

    __host__ __device__ inline float norm() const { return sqrt(dot(*this)); }

    __host__ __device__ inline Vec3d normalized() const { return (*this) / norm(); }

    __host__ __device__ inline void normalize() { (*this) = (*this) / norm(); }
};


__host__ __device__ inline Vec3d operator*(float scalar, const Vec3d &vec) {
    return vec * scalar;
}

#endif // VEC3D_CUH