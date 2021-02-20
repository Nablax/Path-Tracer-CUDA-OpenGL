//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_HITTABLE_LIST_H
#define CUDARAYTRACER_HITTABLE_LIST_H

#include <vector>
#include <memory>
#include "hittable.h"
#include <thrust/device_vector.h>

//class hittable_list : public hittable {
//public:
//    __device__ hittable_list() {}
//    __device__ hittable_list(const std::shared_ptr<hittable>& object) { add(object); }
//
//    __device__ void clear() { objects.clear(); }
//    __device__ void add(const std::shared_ptr<hittable>& object) { objects.push_back(object); }
//
//    __device__ virtual bool hit(
//            const ray& r, double t_min, double t_max, hit_record& rec) const override;
//
//public:
//    thrust::device_vector<std::shared_ptr<hittable>> objects;
//};
//
//__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
//    hit_record temp_rec;
//    bool hit_anything = false;
//    auto closest_so_far = t_max;
//
//    for (int i = 0; i < objects.size(); i++) {
//        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
//            hit_anything = true;
//            closest_so_far = temp_rec.t;
//            rec = temp_rec;
//        }
//    }
//
//    return hit_anything;
//}


#endif //CUDARAYTRACER_HITTABLE_LIST_H
