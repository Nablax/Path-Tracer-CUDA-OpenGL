//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_CAMERA_H
#define RAY_TRACING_CAMERA_H

#include "ray.h"

class camera {
public:
    __host__ __device__ camera(
        point3 lookfrom,
        point3 lookat,
        vec3   vup,
        float vfov, // vertical field-of-view in degrees
        float aspect_ratio,
        float aperture,
        float focus_dist,
        float _time0 = 0,
        float _time1 = 0
    ) {
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta/2);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = vectorgpu::normalize(lookfrom - lookat);
        u = vectorgpu::normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;
        lens_radius = aperture / 2;

        time0 = _time0;
        time1 = _time1;
    }

    __device__ inline ray get_ray(float s, float t, curandState *randState) const {
        vec3 rd = lens_radius * utils::randomInUnitDisk(randState);
        vec3 offset = u * rd.x() + v * rd.y();
        return {origin + offset,
                lower_left_corner + s * horizontal + t * vertical - origin - offset,
                utils::randomInRange(time0, time1, randState)};
    }

public:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    float time0, time1;
};

#endif //RAY_TRACING_CAMERA_H
