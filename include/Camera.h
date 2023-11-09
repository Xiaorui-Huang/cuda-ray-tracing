#ifndef CAMERA_H
#define CAMERA_H

#include "Vec3d.cuh"

struct Camera
{
  // Origin or "eye"
  Vec3d e;
  // orthonormal frame so that -w is the viewing direction. 
  Vec3d u,v,w;
  // image plane distance / focal length
  double d;
  // width and height of image plane
  double width, height;
};

#endif