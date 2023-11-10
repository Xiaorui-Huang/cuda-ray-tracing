#ifndef MATERIAL_H
#define MATERIAL_H

#include "Vec3d.cuh"

// Blinn-Phong Approximate Shading Material Parameters
struct Material
{
  // Ambient, Diffuse, Specular, Mirror Color
  Vec3d ka,kd,ks,km;
  // Phong exponent
  float phong_exponent;
};
#endif
