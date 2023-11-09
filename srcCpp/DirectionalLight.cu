#include "DirectionalLight.h"
#include <limits>

void DirectionalLight::direction(
  const Vec3d & q, Vec3d & d, double & max_t) const
{
    // Recall we are doing backwards tracing
    // from q going the opposite direction of the direction of the light to
    // reach the light.
    d = -this->d;
    max_t = std::numeric_limits<double>::infinity();
}

