//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_CUDA_OBJECT_H
#define CUDARAYTRACER_CUDA_OBJECT_H
#include "aabb.h"
#include "material.h"

#define TYPE_SPHERE 1
#define TYPE_MESH 2

class CudaObj {
public:
    __device__ CudaObj(point3 cen, float r, material* m)
    : mCenter(cen), mRadius(r), mType(TYPE_SPHERE){
        mat_ptr = m;
        mBoundingBox = aabb(mCenter - vec3(mRadius, mRadius, mRadius),
                            mCenter + vec3(mRadius, mRadius, mRadius));
    };
    __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) {
        if(mType == TYPE_SPHERE){
            vec3 oc = r.origin() - mCenter;
            float a = r.direction().length_squared();
            float half_b = dot(oc, r.direction());
            float c = oc.length_squared() - mRadius * mRadius;

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
            vec3 outward_normal = (rec.p - mCenter) / mRadius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        return false;
    }
    __device__ bool bounding_box(float time0, float time1, aabb& output_box) {
        if(mType == TYPE_SPHERE){
            output_box = aabb(mCenter - vec3(mRadius, mRadius, mRadius),
                              mCenter + vec3(mRadius, mRadius, mRadius));
        }
        return false;
    }
    material *mat_ptr = nullptr;
    aabb mBoundingBox;
    int mType = TYPE_SPHERE;
    point3 mCenter;
    float mRadius = 0;
};

#endif //CUDARAYTRACER_CUDA_OBJECT_H
