//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_HITTABLE_H
#define CUDARAYTRACER_HITTABLE_H
#include "aabb.h"
#include "material.h"

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
    __device__ virtual ~hittable() {};
    material *mat_ptr = nullptr;
    aabb mBoundingBox;
};

#endif //CUDARAYTRACER_HITTABLE_H
