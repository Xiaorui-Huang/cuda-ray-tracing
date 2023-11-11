#include <stdio.h>
#include "generate_ray.cuh"

__device__ void generate_ray(
    const Camera &camera, const int i, const int j, const int width, const int height, Ray &ray) {
    // Assumeing camera.u .v .w are all unit vectors
    float px_width = camera.width / width, px_height = camera.height / height;

    // start from the camera
    ray.origin = camera.e;

    ray.direction = (camera.d * -camera.w) + // use -camera.v since v is up and we index downwards
                                             // j + 0.5 and i + 0.5 is to direct the ray at the
                                             // centre of the pixel, instead of the top left corner
                    ((i + 0.5) * px_height * -camera.v) + ((j + 0.5) * px_width * camera.u) +
                    // move the ray to the top left corner of the plane
                    (camera.height / 2 * camera.v - camera.width / 2 * camera.u);
}
