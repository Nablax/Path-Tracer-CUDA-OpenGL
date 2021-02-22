//
// Created by LZR on 2021/2/21.
//

#ifndef CUDARAYTRACER_MATERIAL_H
#define CUDARAYTRACER_MATERIAL_H

#include "physical.h"
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
        scattered = ray(rec.p, scatter_direction, r_in.time());
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
        vec3 reflected = rayphysics::reflect(vectorgpu::normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * utils::randomInUnitSphereDiscard(randState), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

public:
    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__
    dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__
    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *randState
    ) const override {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

        vec3 unit_direction = vectorgpu::normalize(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || rayphysics::reflectance(cos_theta, refraction_ratio) > curand_uniform(randState))
            direction = rayphysics::reflect(unit_direction, rec.normal);
        else
            direction = rayphysics::refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction, r_in.time());

        return true;
    }

public:
    float ir; // Index of Refraction

};


#endif //CUDARAYTRACER_MATERIAL_H
