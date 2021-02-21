//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_HITTABLE_LIST_H
#define CUDARAYTRACER_HITTABLE_LIST_H

#include "hittable.h"

class hittable_list{
public:
    __device__ hittable_list() = default;
    __device__ hittable_list(size_t sz){
        initObj(sz);
    }
    __device__ void clear() {
        delete[] *objects;
        delete objects;
    }
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ void initObj(size_t sz){
        objects = new hittable*[2];
        objSize = sz;
    }
    __device__ ~hittable_list(){
        clear();
    }
public:
    hittable **objects;
    size_t objSize = 0;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < objSize; i++) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}


#endif //CUDARAYTRACER_HITTABLE_LIST_H
