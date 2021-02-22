//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_RAY_H
#define CUDARAYTRACER_RAY_H

class ray {
public:
    __device__ ray() {}
    __device__ ray(const point3& origin, const vec3& direction, float time = 0.0f):
    orig(origin), dir(direction), tm(time){}

    __device__ point3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }
    __device__ float time() const {return tm;};

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }
public:
    point3 orig;
    vec3 dir;
    float tm;
};

#endif //CUDARAYTRACER_RAY_H
