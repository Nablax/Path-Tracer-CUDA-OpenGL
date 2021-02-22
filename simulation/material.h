//
// Created by LZR on 2021/2/21.
//

#ifndef CUDARAYTRACER_MATERIAL_H
#define CUDARAYTRACER_MATERIAL_H

#include "hittable.h"

struct hit_record;

class material {
public:
    __device__
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState
    ) const = 0;

    __device__
    virtual ~material(){}
};

class lambertian : public material {
public:
    __device__
    lambertian(const color& a) : albedo(a) {}

    __device__
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState
    ) const override {
        vec3 scatter_direction = rec.normal + utils::randomOnUnitSphereDiscard(randState);
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

class metal : public material {
public:
    __device__
    metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__
    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState
    ) const override {
        vec3 reflected = reflect(vectorgpu::normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * utils::randomInUnitSphereDiscard(randState));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

public:
    color albedo;
    float fuzz;
};

#endif //CUDARAYTRACER_MATERIAL_H
