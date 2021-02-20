//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_UTILITY_H
#define RAY_TRACING_UTILITY_H

#include <cstdlib>
#include <limits>

#include <random>
#include "vec3.h"
#include "global_variables.h"

using namespace globalvar;

namespace utils{
// Usings
    using color = vectorgpu::vec3;
    using point3 = vectorgpu::vec3;
    using vec3 = vectorgpu::vec3;

//    using std::shared_ptr;
//    using std::make_shared;
//    using std::sqrt;

// Utility Functions

    inline float degrees_to_radians(float degrees) {
        return degrees * kDegToRad;
    }

    inline float clamp(float x, float min, float max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    inline float random_float() {
        static std::uniform_real_distribution<float> distribution(0.0, 1.0);
        static std::mt19937 generator;
        return distribution(generator);
    }

    inline float random_float(float min, float max) {
        // Returns a random real in [min,max).
        return min + (max-min)*random_float();
    }

//    inline vec3 random_vec3(){
//        return {random_float(), random_float(), random_float()};
//    }
//
//    inline vec3 random_vec3(float min, float max){
//        return {random_float(min,max), random_float(min,max), random_float(min,max)};
//    }

//    vec3 random_in_unit_sphere() {
//        while (true) {
//            auto p = random_vec3(-1,1);
//            if (vectorgpu::dot(p , p) >= 1) continue;
//            return p;
//        }
//    }

//    vec3 random_unit_vector() {
//        auto a = random_float(0, 2* kPi);
//        auto z = random_float(-1, 1);
//        auto r = sqrt(1 - z*z);
//        return {r*cos(a), r*sin(a), z};
//    }

//    vec3 random_in_hemisphere(const vec3& normal) {
//        vec3 in_unit_sphere = random_in_unit_sphere();
//        if (vectorgpu::dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//            return in_unit_sphere;
//        else
//            return -in_unit_sphere;
//    }

    vec3 reflect(const vec3& v, const vec3& n) {
        return v - n * vectorgpu::dot(v,n)*2;
    }

    vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
        auto cos_theta = vectorgpu::dot(-uv, n);
        vec3 r_out_perp =  etai_over_etat * (uv + n * cos_theta);
        vec3 r_out_parallel = -sqrt(fabs(1.0 - vectorgpu::dot(r_out_perp, r_out_perp))) * n;
        return r_out_perp + r_out_parallel;
    }

    float schlick(float cosine, float ref_idx) {
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }

    vec3 random_in_unit_disk() {
        while (true) {
            auto p = vec3(random_float(-1,1), random_float(-1,1), 0);
            if (vectorgpu::dot(p, p) >= 1) continue;
            return p;
        }
    }
}

#endif //RAY_TRACING_UTILITY_H
