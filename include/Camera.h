#ifndef CAMERA_H
#define CAMERA_H

#include "Float3d.cuh"

struct Camera
{
  // Origin or "eye"
  float3d e;
  // orthonormal frame so that -w is the viewing direction. 
  float3d u,v,w;
  // image plane distance / focal length
  float d;
  // width and height of image plane
  float width, height;
};

#endif
