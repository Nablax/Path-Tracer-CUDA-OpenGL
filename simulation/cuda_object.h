//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_CUDA_OBJECT_H
#define CUDARAYTRACER_CUDA_OBJECT_H
#include "aabb.h"
#include "OBJ_Loader.hpp"
#include "triangle.h"

#define TYPE_SPHERE 1
#define TYPE_MESH 2

class CudaObj {
public:
    __host__ __device__
    CudaObj(){}
    __host__ __device__
    CudaObj(point3 cen, float r, int matID)
    : mCenter(cen), mRadius(r), mMaterialID(matID), mType(TYPE_SPHERE){
        if(mType == TYPE_SPHERE){
            r = fabsf(r);
            mBoundingBox = aabb(mCenter - vec3(r, r, r),
                                mCenter + vec3(r, r, r));
        }
    };
    __host__
    CudaObj(const std::string& fileName, int matID): mMaterialID(matID), mType(TYPE_MESH){
        objl::Loader loader;
        loader.LoadFile(fileName);

    }
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
            getUV(outward_normal, rec.u, rec.v);
            rec.matID = mMaterialID;
            return true;
        }
        return false;
    }
    __device__ void getUV(const point3 &p, float &u, float &v){
        if(mType == TYPE_SPHERE){
            float theta = acosf(-p.y());
            float phi = atan2f(-p.z(), p.x()) + globalvar::kPiGPU;

            u = phi * 0.5 * globalvar::kPiGPUInv;
            v = theta * globalvar::kPiGPUInv;
        }
    }

    __device__ bool bounding_box(float time0, float time1, aabb& output_box) {
        if(mType == TYPE_SPHERE){
            float r = fabsf(mRadius);
            output_box = aabb(mCenter - vec3(r, r, r),
                              mCenter + vec3(r, r, r));
        }
        return false;
    }
    triangle *mTriangles;
    int mTriCount = 0;
    int mMaterialID = -1;
    aabb mBoundingBox;
    int mType = TYPE_SPHERE;
    point3 mCenter;
    float mRadius = 0;
};

#endif //CUDARAYTRACER_CUDA_OBJECT_H
