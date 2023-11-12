#ifndef GENERATE_RAY_CUH
#define GENERATE_RAY_CUH

#include "Camera.h"
#include "Ray.h"

/**
 * @brief Construct a viewing ray given a camera and subscripts to a pixel.
 *
 * @pre Camera .u .v. w are all unit vectors.
 *
 * @param camera Perspective camera object.
 * @param i Pixel row index.
 * @param j Pixel column index.
 * @param width Number of pixels width of image.
 * @param height Number of pixels height of image.
 * @return ray - Viewing ray starting at camera, shooting through pixel. When `t=1`, then
 * `ray.origin + t*ray.direction` should land exactly on the center of the pixel `(i,j)`.
 */
__device__ Ray generate_ray(
    const Camera &camera, const int i, const int j, const int width, const int height);

#endif
