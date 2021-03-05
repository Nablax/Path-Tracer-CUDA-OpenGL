//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_BVH_H
#define CUDARAYTRACER_BVH_H

#include "cuda_object.h"
#include "render_manager.h"

class bvh_node{
public:
    __device__
    bvh_node(){};

    __device__
    bvh_node(const RenderManager* list, float time0, float time1)
            : bvh_node(list->objects, 0, list->objLastIdx, time0, time1){}

    __device__
    bvh_node(CudaObj **src_objects,
             size_t start, size_t end, float time0, float time1);

    __device__
    bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;

    __device__
    bool bounding_box(float time0, float time1, aabb& output_box) const;

public:
    CudaObj *left, *right;
    aabb box;
};

__device__
inline bool bvh_node::bounding_box(float time0, float time1, aabb &output_box) const {
    return false;
}

__device__
inline bool bvh_node::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__
inline bvh_node::bvh_node(CudaObj **src_objects, size_t start, size_t end, float time0, float time1) {

}


#endif //CUDARAYTRACER_BVH_H
