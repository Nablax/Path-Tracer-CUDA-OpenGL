//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_RENDER_MANAGER_H
#define CUDARAYTRACER_RENDER_MANAGER_H

#include "cuda_object.h"
#include "material.h"
#include "aabb.h"

class RenderManager{
public:
    __device__ RenderManager() {};
    __device__ RenderManager(size_t objSz, size_t matSz){
        initObj(objSz);
        initMat(matSz);
    }
    __device__ inline void clear() {
        for(int i = 0; i < objLastIdx; i++){
            delete objects[i];
        }
        delete objects;

        for(int i = 0; i < matLastIdx; i++){
            delete mats[i];
        }
        delete mats;
    }
    __device__ inline void addObj(CudaObj *o){
        if(objMaxSize <= 0) return;
        objLastIdx %= objMaxSize;
        objects[objLastIdx++] = o;
        mWorldBoundingBox.unionBoxInPlace(o->mBoundingBox);
    }
    __device__ inline void addMat(material *m){
        if(matMaxSize <= 0) return;
        matLastIdx %= matMaxSize;
        mats[matLastIdx++] = m;
    }
    __device__ inline bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__
    bool unionAllBox(float time0, float time1, aabb& output_box) const ;
    __device__ inline void initObj(size_t sz){
        objects = new CudaObj*[sz];
        objLastIdx = 0;
        objMaxSize = sz;
        mWorldBoundingBox.mMin = mWorldBoundingBox.mMax = vec3();
    }
    __device__ inline void initMat(size_t sz){
        mats = new material*[sz];
        matLastIdx = 0;
        matMaxSize = sz;
    }
    __device__ inline ~RenderManager(){
        clear();
    }
public:
    CudaObj **objects;
    material **mats;
    aabb mWorldBoundingBox;
    size_t objLastIdx = 0;
    size_t matLastIdx = 0;
    size_t matMaxSize = 0;
    size_t objMaxSize = 0;
};

__device__ inline bool RenderManager::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < objLastIdx; i++) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool RenderManager::unionAllBox(float time0, float time1, aabb &output_box) const {
    if(objLastIdx <= 0) return false;
    aabb tmpBox;
    if(!objects[0]->bounding_box(time0, time1, tmpBox)) return false;
    output_box = tmpBox;
    for(int i = 1; i < objLastIdx; i++){
        if(!objects[0]->bounding_box(time0, time1, tmpBox)) return false;
        output_box = utils::unionBox(output_box, tmpBox);
    }
    return true;
}


#endif //CUDARAYTRACER_RENDER_MANAGER_H
