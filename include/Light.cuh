#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "Float3d.cuh"

enum class LightType { DirectionalLight, PointLight };

struct DirectionalLight {
    float3d direction;
};

struct PointLight {
    float3d position;
};

union LightData {
    DirectionalLight directional;
    PointLight point;
};

// Define the Light struct that holds the type and the union
struct Light {
    LightType type;
    LightData data;
    // Intensity of the light
    float3d color; 

    __host__ __device__ Light(const DirectionalLight &direction, const float3d &color)
        : type(LightType::DirectionalLight), data({.directional = direction}), color(color) {}

    __host__ __device__ Light(const PointLight &position, const float3d &color)
        : type(LightType::PointLight), data({.point = position}), color(color) {}

    // Given a query point return the direction _toward_ the Light.
    //
    // Input:
    //   query  3D query point in space
    // Outputs:
    //    direction  3D direction from point toward light as a vector.
    //    maxT  parametric distance from q along d to light (may be inf)
    __device__ void direction(const float3d &query, float3d &direction, float &maxT) const {
        switch (type) {
        case LightType::DirectionalLight:
            direction = -data.directional.direction;
            maxT = INFINITY; // Directional light is at infinity
            break;
        case LightType::PointLight:
            direction = data.point.position - query;
            maxT = 1;
            // here out implementation is not to use a unit direction vector

            // maxT = direction.norm();           // Distance to the point light
            // direction = direction.normalized(); // Normalize the direction
            break;
        }
    }
};

#endif // CUDA_LIGHT_H
