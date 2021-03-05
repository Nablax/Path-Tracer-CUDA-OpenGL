//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_SPHERE_H
#define CUDARAYTRACER_SPHERE_H

#include "material.h"
#include "aabb.h"

class sphere : public hittable{
public:
    __device__ sphere() {}
    __device__ sphere(point3 cen, float r, material* m)
            : center(cen), radius(r){
        mat_ptr = m;
        mBoundingBox = aabb(center - vec3(radius, radius, radius),
                            center + vec3(radius, radius, radius));
    };
    __device__ virtual inline bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__
    virtual inline bool bounding_box(float time0, float time1, aabb& output_box) const override;
public:
    point3 center;
    float radius;
};

__device__ inline bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrt = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrt) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrt) / a;
        if (root < t_min || t_max < root)
            return false;
    }
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ inline bool sphere::bounding_box(float time0, float time1, aabb &output_box) const {
    output_box = aabb(center - vec3(radius, radius, radius),
                      center + vec3(radius, radius, radius));
    return false;
}

#endif //CUDARAYTRACER_SPHERE_H
