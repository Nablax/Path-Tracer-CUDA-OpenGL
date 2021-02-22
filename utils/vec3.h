//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_VEC3_H
#define CUDARAYTRACER_VEC3_H
#include <cmath>
#include <iostream>
namespace vectorgpu{
    class vec3{
        public:
        float e[4];
        __host__ __device__ inline vec3():e{0, 0, 0}{}
        __host__ __device__ inline vec3(float e1, float e2, float e3, float e4 = 0): e{e1, e2, e3}{}
        __host__ __device__ inline float x() const { return e[0];};
        __host__ __device__ inline float y() const { return e[1];};
        __host__ __device__ inline float z() const { return e[2];};
        __host__ __device__ inline float w() const { return e[3];};
        __host__ __device__ inline float r() const { return e[0];};
        __host__ __device__ inline float g() const { return e[1];};
        __host__ __device__ inline float b() const { return e[2];};
        __host__ __device__ inline float a() const { return e[3];};
        __host__ __device__ inline vec3 operator-() const { return {-e[0], -e[1], -e[2], e[3]};}
        __host__ __device__ inline vec3& operator+=(const vec3 &inVec) {
            e[0] += inVec.e[0];
            e[1] += inVec.e[1];
            e[2] += inVec.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator-=(const vec3 &inVec) {
            e[0] -= inVec.e[0];
            e[1] -= inVec.e[1];
            e[2] -= inVec.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator*=(const vec3 &inVec) {
            e[0] *= inVec.x();
            e[1] *= inVec.y();
            e[2] *= inVec.z();
            return *this;
        }
        __host__ __device__ inline vec3& operator*=(const float &t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }
        __host__ __device__ inline vec3& operator/=(const float &t) {
            return *this *= (1 / t);
        }
        __host__ __device__ inline float length_squared() const {return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];}
        __host__ __device__ inline float length() const {return std::sqrt(length_squared());}
        __host__ __device__ inline vec3& normalized() {
            float tmpLen = this->length();
            if(tmpLen == 0) return *this;
            return *this /= tmpLen;
        }
        __host__ __device__ inline bool near_zero() const {
            const float s = 1e-7;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }
    };
    inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
        return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
    }
    __host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
        return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
    }
    __host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
        return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
    }
    __host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
        return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
    }
    __host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
        return {t * v.e[0], t * v.e[1], t * v.e[2]};
    }
    __host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
        return t * v;
    }
    __host__ __device__ inline vec3 operator/(vec3 v, float t) {
        return (1 / t) * v;
    }
    __host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
        return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
    }
    __host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
        return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]};
    }
    __host__ __device__ inline vec3 normalize(vec3 v) {
        float tmpLen = v.length();
        if(tmpLen == 0) return {};
        return v / tmpLen;
    }
    __host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2 * dot(v, n) * n;
    }
}

#endif //CUDARAYTRACER_VEC3_H
