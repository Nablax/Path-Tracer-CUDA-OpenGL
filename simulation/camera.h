//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_CAMERA_H
#define RAY_TRACING_CAMERA_H

#include "ray.h"

class camera {
public:
    camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            float vfov, // vertical field-of-view in degrees
            float aspect_ratio,
            float aperture,
            float focus_dist
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = glm::normalize(lookfrom - lookat);
        u = glm::normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        higher_left_corner = origin - horizontal * 0.5 + vertical * 0.5 - focus_dist*w;

        lens_radius = aperture / 2;
    }


    ray get_ray(float s, float t) const {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x + v * rd.y;

        return ray(
                origin + offset,
                higher_left_corner + s*horizontal - t*vertical - origin - offset
        );
    }

private:
    point3 origin;
    point3 higher_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif //RAY_TRACING_CAMERA_H
