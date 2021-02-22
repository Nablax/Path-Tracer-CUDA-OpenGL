//
// Created by LZR on 2021/2/21.
//

#ifndef CUDARAYTRACER_PHYSICAL_H
#define CUDARAYTRACER_PHYSICAL_H

#include "vec3.h"

namespace rayphysics{
    __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2 * dot(v, n) * n;
    }
    __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
        float cos_theta = fminf(dot(-uv, n), 1.0);
        vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
        vec3 r_out_parallel = -sqrtf(fabs(1.0f - r_out_perp.length_squared())) * n;
        return r_out_perp + r_out_parallel;
    }
    __device__ inline float reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * powf((1 - cosine),5);
    }
}

#endif //CUDARAYTRACER_PHYSICAL_H
