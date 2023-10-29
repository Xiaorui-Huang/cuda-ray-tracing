#include "PointLight.h"

void PointLight::direction(
  const Vec3d & q, Vec3d & d, double & max_t) const
{
    // Recall we are doing backwards tracing - from query q back to light source p
    d = this->p - q;
    max_t = 1;
}
