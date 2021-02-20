//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_HITTABLE_H
#define CUDARAYTRACER_HITTABLE_H

struct hit_record {
    point3 p;
    vec3 normal;
    float t = 0;
    bool front_face = false;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif //CUDARAYTRACER_HITTABLE_H
