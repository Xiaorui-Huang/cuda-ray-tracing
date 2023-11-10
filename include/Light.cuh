#ifndef CUDA_LIGHT_H
#define CUDA_LIGHT_H

#include "Vec3d.cuh"

enum class LightType { DirectionalLight, PointLight };

struct DirectionalLight {
    Vec3d direction;
};

struct PointLight {
    Vec3d position;
};

union LightData {
    DirectionalLight directional;
    PointLight point;
};

// Define the Light struct that holds the type and the union
struct Light {
    LightType type;
    LightData data;

    __host__ __device__ Light(const Vec3d &direction)
        : type(LightType::DirectionalLight), data({.directional = direction}) {}

    __host__ __device__ Light(const Vec3d &position)
        : type(LightType::PointLight), data({.point = position}) {}

    // Given a query point return the direction _toward_ the Light.
    //
    // Input:
    //   query  3D query point in space
    // Outputs:
    //    direction  3D direction from point toward light as a vector.
    //    maxT  parametric distance from q along d to light (may be inf)
    __device__ void direction(const Vec3d &query, Vec3d &direction, double &maxT) const {
        switch (type) {
        case LightType::DirectionalLight:
            direction = data.directional.direction;
            maxT = INFINITY; // Directional light is at infinity
            break;
        case LightType::PointLight:
            direction = data.point.position - query;
            maxT = direction.norm();           // Distance to the point light
            direction = direction.normalized(); // Normalize the direction
            break;
        }
    }
};

#endif // CUDA_LIGHT_H
