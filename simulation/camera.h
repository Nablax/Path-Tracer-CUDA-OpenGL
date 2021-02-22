//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_CAMERA_H
#define RAY_TRACING_CAMERA_H

#include "ray.h"

class camera {
public:
    __host__ __device__ camera() {
        float viewport_height = 2.0;
        float viewport_width = globalvar::kAspectRatio * viewport_height;
        float focal_length = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewport_width, 0.0, 0.0);
        vertical = vec3(0.0, viewport_height, 0.0);
        lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
    }

    __device__ ray get_ray(float u, float v) const {
        return {origin, lower_left_corner + u*horizontal + v*vertical - origin};
    }

public:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

#endif //RAY_TRACING_CAMERA_H
