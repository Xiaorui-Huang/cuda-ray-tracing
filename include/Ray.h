#ifndef RAY_H
#define RAY_H

#include "Float3d.cuh"

struct Ray 
{
  float3d origin;
  // Not necessarily unit-length direction vector. (It is often useful to have
  // non-unit length so that origin+t*direction lands on a special point when
  // t=1.)
  float3d direction;
};

#endif
