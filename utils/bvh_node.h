//
// Created by LZR on 2021/3/14.
//

#ifndef CUDARAYTRACER_BVH_NODE_H
#define CUDARAYTRACER_BVH_NODE_H

struct BVHNode{
    int left = -1, right = -1;
    int parent = -1;
    int objID = -1;
    aabb box;
    __device__
    bool isLeafNode(){
        return objID != -1;
    }
};

#endif //CUDARAYTRACER_BVH_NODE_H
