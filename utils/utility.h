//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_UTILITY_H
#define RAY_TRACING_UTILITY_H

#include <curand_kernel.h>
#include "vec3.h"
#include "global_variables.h"
#include <random>

namespace utils{
    using color = vectorgpu::vec3;
    using point3 = vectorgpu::vec3;
    using vec3 = vectorgpu::vec3;

    enum directions{
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT
    };

    __host__ __device__
    inline float degrees_to_radians(float degrees) {
        return degrees * globalvar::kDegToRadGPU;
    }

    template<typename T>
    __device__
    inline void swapGPU(T &a, T &b){
        T tmp = a;
        a = b;
        b = tmp;
    }

    inline float clamp(float x, float min, float max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __device__ inline float randomInRange(float left, float right, curandState *randState){
        if(right <= left) return 0;
        return curand_uniform(randState) * (right - left) + left;
    }

    __device__ inline vec3 randomOnUnitSphereDiscard(curandState *randState){
        vec3 res;
        float norm = 1;
        do{
            res = 2 * vec3(
                    curand_uniform(randState) - 0.5f,
                    curand_uniform(randState) - 0.5f,
                    curand_uniform(randState) - 0.5f);
            norm = res.length_squared();
        }while(res.length_squared() >= 1.0f);
        return res / sqrtf(norm);
    }

    __device__ inline vec3 randomInUnitSphereCbrt(curandState *randState){
        vec3 res = vec3(
                curand_uniform(randState) - 0.5f,
                curand_uniform(randState) - 0.5f,
                curand_uniform(randState) - 0.5f);
        float u = cbrtf(curand_uniform(randState));
        return res.normalized() * u;
    }

    __device__ inline vec3 randomInUnitSphereDiscard(curandState *randState){
        vec3 res;
        do{
            res = 2 * vec3(
                    curand_uniform(randState) - 0.5f,
                    curand_uniform(randState) - 0.5f,
                    curand_uniform(randState) - 0.5f);
        }while(res.length_squared() >= 1.0f);
        return res;
    }

    __device__ inline vec3 randomOnUnitSphere(curandState *randState){
        float phi = curand_uniform(randState) * 2.0f * globalvar::kPiGPU;
        float cosTheta = 1.0f - 2 * curand_uniform(randState);
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
    }

    __device__ inline vec3 randomInUnitHemisphere(curandState *randState, const vec3 &normal){
        vec3 inUnitSphere = randomInUnitSphereDiscard(randState);
        if(vectorgpu::dot(inUnitSphere, normal) > 0)
            return inUnitSphere;
        return -inUnitSphere;
    }

    __device__ inline vec3 randomInUnitDisk(curandState *randState){
        float r = sqrtf(curand_uniform(randState));
        float theta = curand_uniform(randState) * 2 * globalvar::kPiGPU;
        return {r * cosf(theta), r * sinf(theta), 0};
    }
    __host__
    inline float randomUniformOnHost() {
        static std::uniform_real_distribution<float> distribution(0.0, 1.0);
        static std::mt19937 generator;
        return distribution(generator);
    }
}

#endif //RAY_TRACING_UTILITY_H
