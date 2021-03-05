//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_RAY_H
#define CUDARAYTRACER_RAY_H

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const point3& origin, const vec3& direction, float time = 0.0f):
            mOrigin(origin), mDir(direction), mTime(time){}

    __device__ point3 origin() const { return mOrigin; }
    __device__ vec3 direction() const { return mDir; }
    __device__ float time() const {return mTime;};

    __device__ point3 at(float t) const {
        return mOrigin + t * mDir;
    }
private:
    point3 mOrigin;
    vec3 mDir;
    float mTime;
};

#endif //CUDARAYTRACER_RAY_H
