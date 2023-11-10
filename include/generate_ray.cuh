#ifndef VIEWING_RAY_CUH
#define VIEWING_RAY_CUH

#include "Camera.h"
#include "Ray.h"

/**
 * @brief Construct a viewing ray given a camera and subscripts to a pixel. 
 * 
 * @pre Camera .u .v. w are all unit vectors.
 *
 * @param camera `Input` - Perspective camera object.
 * @param i `Input` - Pixel row index.
 * @param j `Input` - Pixel column index.
 * @param width `Input` - Number of pixels width of image.
 * @param height `Input` - Number of pixels height of image.
 * @param ray `Output` - Viewing ray starting at camera, shooting through pixel. When `t=1`, then
 * `ray.origin + t*ray.direction` should land exactly on the center of the pixel `(i,j)`.
 */
__device__ void generateRay(
    const Camera &camera, const int i, const int j, const int width, const int height, Ray &ray);

#endif
