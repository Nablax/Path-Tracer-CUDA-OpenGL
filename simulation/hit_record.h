//
// Created by LZR on 2021/3/2.
//

#ifndef CUDARAYTRACER_HIT_RECORD_H
#define CUDARAYTRACER_HIT_RECORD_H

#include "material.h"

class Material;

struct hit_record {
    point3 p;
    vec3 normal;
    int matID = -1;
    float t = 0;
    bool front_face = false;

    __device__
    inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};


#endif //CUDARAYTRACER_HIT_RECORD_H
