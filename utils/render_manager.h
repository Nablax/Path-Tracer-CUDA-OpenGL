//
// Created by LZR on 2021/2/20.
//

#ifndef CUDARAYTRACER_RENDER_MANAGER_H
#define CUDARAYTRACER_RENDER_MANAGER_H

#include "material.h"
#include "bvh.h"

class RenderManager{
public:
    __device__ RenderManager() {};
    __device__ RenderManager(size_t objSz, size_t matSz){
        initObj(objSz);
        initMat(matSz);
    }
    __device__ inline void clear() {
        delete[] mObjects;
        delete[] mMaterials;
    }
    __device__ inline void addObj(CudaObj *o){
        if(mObjMaxSize <= 0) return;
        mObjLastIdx %= mObjMaxSize;
        mObjects[mObjLastIdx++] = *o;
        mWorldBoundingBox.unionBoxInPlace(o->mBoundingBox);
    }
    __device__ inline void addMat(Material *m){
        if(mMatMaxSize <= 0) return;
        mMatLastIdx %= mMatMaxSize;
        mMaterials[mMatLastIdx++] = *m;
    }
    __device__ inline bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ inline bool hitBvh(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__
    bool unionAllBox(float time0, float time1, aabb& output_box) const ;
    __device__ inline void initObj(size_t sz){
        mObjects = new CudaObj[sz];
        mObjLastIdx = 0;
        mObjMaxSize = sz;
        mWorldBoundingBox.mMin = mWorldBoundingBox.mMax = vec3();
    }
    __device__ inline void initMat(size_t sz){
        mMaterials = new Material[sz];
        mMatLastIdx = 0;
        mMatMaxSize = sz;
    }
    __device__ inline ~RenderManager(){
        clear();
    }
    __device__ inline void printBvh(){
        for(int i = 0; i < 2 * mObjMaxSize - 1; i++){
            point3 tmpMin = bvh[i].box.mMin;
            point3 tmpMax = bvh[i].box.mMax;
            printf("Node: %d, ObjID: %d, parent, left, right: %d, %d, %d, minIdx: %.2f %.2f %.2f, maxIdx: %.2f %.2f %.2f\n",
                   i, bvh[i].objID, bvh[i].parent, bvh[i].left, bvh[i].right, tmpMin.x(), tmpMin.y(), tmpMin.z(),
                   tmpMax.x(), tmpMax.y(), tmpMax.z());
        }
    }
public:
    CudaObj *mObjects;
    Material *mMaterials;
    aabb mWorldBoundingBox;
    BVHNode *bvh;
    size_t mObjLastIdx = 0;
    size_t mMatLastIdx = 0;
    size_t mMatMaxSize = 0;
    size_t mObjMaxSize = 0;
};

__device__ inline bool RenderManager::hit(const Ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < mObjLastIdx; i++) {
        if (mObjects[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool RenderManager::hitBvh(const Ray &r, float t_min, float t_max, hit_record &rec) const {
    float closestSoFar = t_max;
    bool hitAnything = false;
    BVHNode curNode = bvh[0];

    hit_record tmpRec;
    if(curNode.isLeafNode()){
        if (mObjects[curNode.objID].hit(r, t_min, closestSoFar, tmpRec)) {
            hitAnything = true;
            rec = tmpRec;
        }
        return hitAnything;
    }

    const int stackNum = 64;
    int queryStack[stackNum];
    queryStack[0] = 0;
    int stackTop = 1;

    while(stackTop > 0){
        curNode = bvh[queryStack[--stackTop]];
        BVHNode nextNode = bvh[curNode.left];
        if(nextNode.box.hit(r, t_min, closestSoFar)){
            if(nextNode.isLeafNode()){
                if (mObjects[nextNode.objID].hit(r, t_min, closestSoFar, tmpRec)) {
                    hitAnything = true;
                    closestSoFar = tmpRec.t;
                    rec = tmpRec;
                }
            }
            else{
                queryStack[stackTop++] = curNode.left;
            }
        }
        nextNode = bvh[curNode.right];
        if(nextNode.box.hit(r, t_min, closestSoFar)){
            if(nextNode.isLeafNode()){
                if (mObjects[nextNode.objID].hit(r, t_min, closestSoFar, tmpRec)) {
                    hitAnything = true;
                    closestSoFar = tmpRec.t;
                    rec = tmpRec;
                }
            }
            else{
                queryStack[stackTop++] = curNode.right;
            }
        }
    }
    return hitAnything;
}

__device__ bool RenderManager::unionAllBox(float time0, float time1, aabb &output_box) const {
    if(mObjLastIdx <= 0) return false;
    aabb tmpBox;
    if(!mObjects[0].bounding_box(time0, time1, tmpBox)) return false;
    output_box = tmpBox;
    for(int i = 1; i < mObjLastIdx; i++){
        if(!mObjects[0].bounding_box(time0, time1, tmpBox)) return false;
        output_box = utils::unionBox(output_box, tmpBox);
    }
    return true;
}


#endif //CUDARAYTRACER_RENDER_MANAGER_H
