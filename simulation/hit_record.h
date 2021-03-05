//
// Created by LZR on 2021/3/2.
//

#ifndef CUDARAYTRACER_HIT_RECORD_H
#define CUDARAYTRACER_HIT_RECORD_H

#include "material.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    material *mat_ptr = nullptr;
    float t = 0;
    bool front_face = false;

    __device__
    inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};


#endif //CUDARAYTRACER_HIT_RECORD_H
