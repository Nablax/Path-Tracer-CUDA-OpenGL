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
    float u = 0, v = 0;
    bool front_face = false;

    __device__
    inline void setFaceNormal(const Ray& r, const vec3& outwardNormal) {
        front_face = dot(r.direction(), outwardNormal) < 0;
        normal = front_face ? outwardNormal :-outwardNormal;
    }
};


#endif //CUDARAYTRACER_HIT_RECORD_H
