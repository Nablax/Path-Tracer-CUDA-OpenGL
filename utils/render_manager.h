//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_RENDER_MANAGER_H
#define CUDARAYTRACER_RENDER_MANAGER_H

#include "hittable.h"
#include "material.h"

class RenderManager{
public:
    __device__ RenderManager() {};
    __device__ RenderManager(size_t objSz, size_t matSz){
        initObj(objSz);
        initMat(matSz);
    }
    __device__ void clear() {
        for(int i = 0; i < objLastIdx; i++){
            delete objects[i];
        }
        delete objects;

        for(int i = 0; i < matLastIdx; i++){
            delete mats[i];
        }
        delete mats;
    }
    __device__ void addObj(hittable *o){
        if(objMaxSize <= 0) return;
        objLastIdx %= objMaxSize;
        objects[objLastIdx++] = o;
    }
    __device__ void addMat(material *m){
        if(matMaxSize <= 0) return;
        matLastIdx %= matMaxSize;
        mats[matLastIdx++] = m;
    }
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ void initObj(size_t sz){
        objects = new hittable*[sz];
        objLastIdx = 0;
        objMaxSize = sz;
    }
    __device__ void initMat(size_t sz){
        mats = new material*[sz];
        matLastIdx = 0;
        matMaxSize = sz;
    }
    __device__ ~RenderManager(){
        clear();
    }
public:
    hittable **objects;
    material **mats;
    size_t objLastIdx = 0;
    size_t matLastIdx = 0;
    size_t matMaxSize = 0;
    size_t objMaxSize = 0;
};

__device__ bool RenderManager::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
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


#endif //CUDARAYTRACER_RENDER_MANAGER_H
