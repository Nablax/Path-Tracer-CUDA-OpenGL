//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_CUDA_OBJECT_H
#define CUDARAYTRACER_CUDA_OBJECT_H
#include "aabb.h"
#include "triangle.h"
#include "mesh_loader.h"
#include "triangle.h"

#define TYPE_SPHERE 1
#define TYPE_MESH 2
#define TYPE_SINGLE_TRIANGLE 3

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

    __host__ __device__
    CudaObj(point3 v0, point3 v1, point3 v2, int matID): mMaterialID(matID){
        mTriCount = 1;
        mType = TYPE_SINGLE_TRIANGLE;
        mTriangles = new Triangle(v0, v1, v2);
        point3 tmpMin = v0, tmpMax = v0;
        utils::unionPoints(tmpMin, v1, true);
        utils::unionPoints(tmpMin, v2, true);

        utils::unionPoints(tmpMax, v1, false);
        utils::unionPoints(tmpMax, v2, false);
        mBoundingBox = aabb(tmpMin, tmpMax);
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
            rec.setFaceNormal(r, outward_normal);
            getUV(outward_normal, rec.u, rec.v);
            rec.matID = mMaterialID;
            return true;
        }
        else if(mType == TYPE_SINGLE_TRIANGLE){
            point3 e1 = mTriangles[0].mVertex[1] - mTriangles[0].mVertex[0];
            point3 e2 = mTriangles[0].mVertex[2] - mTriangles[0].mVertex[0];
            point3 s1 = vectorgpu::cross(r.direction(), e2);
            float s1e1inv = vectorgpu::dot(s1, e1);
            if(s1e1inv == 0) return false;
            point3 s = r.origin() - mTriangles[0].mVertex[0];
            point3 s2 = vectorgpu::cross(s, e1);
            s1e1inv = 1.0f / s1e1inv;
            float t = vectorgpu::dot(s2, e2) * s1e1inv;
            float b1 = vectorgpu::dot(s1, s) * s1e1inv;
            float b2 = vectorgpu::dot(s2, r.direction()) * s1e1inv;

            if(b1 >= 1 || b1 <= 0 || b2 >= 1 || b2 <= 0 || b1 + b2 <= 0 || b1 + b2 >= 1|| t <= t_min || t >= t_max)
                return false;
            rec.t = t;
            rec.p = r.at(t);
            rec.setFaceNormal(r, mTriangles[0].mNormal);
            rec.matID = mMaterialID;
            return true;
        }
        return false;
    }

    __device__ inline void getUV(const point3 &p, float &u, float &v){
        if(mType == TYPE_SPHERE){
            float theta = acosf(-p.y());
            float phi = atan2f(-p.z(), p.x()) + globalvar::kPiGPU;

            u = phi * 0.5f * globalvar::kPiGPUInv;
            v = theta * globalvar::kPiGPUInv;
        }
    }

//    __device__ bool bounding_box(float time0, float time1, aabb& output_box) {
//        if(mType == TYPE_SPHERE){
//            float r = fabsf(mRadius);
//            output_box = aabb(mCenter - vec3(r, r, r),
//                              mCenter + vec3(r, r, r));
//        }
//        return false;
//    }
    __host__ __device__
    ~CudaObj(){
        //delete []mTriangles;
    }
    Triangle *mTriangles;
    int mTriCount = 0;
    int mMaterialID = -1;
    aabb mBoundingBox;
    int mType = TYPE_SPHERE;
    point3 mCenter;
    float mRadius = 0;
};

#endif //CUDARAYTRACER_CUDA_OBJECT_H
