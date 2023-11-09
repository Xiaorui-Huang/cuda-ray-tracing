#ifndef RAY_H
#define RAY_H

#include "Vec3d.cuh"

struct Ray 
{
  Vec3d origin;
  // Not necessarily unit-length direction vector. (It is often useful to have
  // non-unit length so that origin+t*direction lands on a special point when
  // t=1.)
  Vec3d direction;
};

#endif
