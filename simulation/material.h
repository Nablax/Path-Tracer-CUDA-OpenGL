//
// Created by LZR on 2021/2/21.
//

#ifndef CUDARAYTRACER_MATERIAL_H
#define CUDARAYTRACER_MATERIAL_H

#include "physical.h"
#include "hit_record.h"

struct hit_record;

#define LAMBERTIAN 1
#define METAL 2
#define DIELECTRIC 4

class material {
public:
    __host__ __device__
    material(const color& a) : mAlbedo(a), mType(LAMBERTIAN) {}
    __host__ __device__
    material(const color& a, float f) : mAlbedo(a), mFuzz(f < 1 ? f : 1), mType(METAL)  {}
    __host__ __device__
    material(const float index_of_refraction) : mIr(index_of_refraction), mType(DIELECTRIC) {}
    __device__
    bool scatter(
            const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState *randState
    ) {
        if(mType == LAMBERTIAN){
            vec3 scatter_direction = rec.normal + utils::randomOnUnitSphereDiscard(randState);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            scattered = Ray(rec.p, scatter_direction, r_in.time());
            attenuation = mAlbedo;
            return true;
        }
        if(mType == METAL){
            vec3 reflected = rayphysics::reflect(vectorgpu::normalize(r_in.direction()), rec.normal);
            scattered = Ray(rec.p, reflected + mFuzz * utils::randomInUnitSphereDiscard(randState), r_in.time());
            attenuation = mAlbedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        if(mType == DIELECTRIC){
            attenuation = color(1.0, 1.0, 1.0);
            float refraction_ratio = rec.front_face ? (1.0f / mIr) : mIr;
            vec3 unit_direction = vectorgpu::normalize(r_in.direction());
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
            float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || rayphysics::reflectance(cos_theta, refraction_ratio) > curand_uniform(randState))
                direction = rayphysics::reflect(unit_direction, rec.normal);
            else
                direction = rayphysics::refract(unit_direction, rec.normal, refraction_ratio);
            scattered = Ray(rec.p, direction, r_in.time());
            return true;
        }
        return false;
    }
public:
    color mAlbedo;
    float mFuzz = 0;
    float mIr = 0;
    int mType = LAMBERTIAN;
};


#endif //CUDARAYTRACER_MATERIAL_H
