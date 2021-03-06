//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_BVH_H
#define CUDARAYTRACER_BVH_H

#include "cuda_object.h"
#include <vector>
#include <algorithm>
#include "aabb.h"

struct Morton{
    unsigned int mortonCodes;
    int objectID;
};

__device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__
unsigned int morton3D(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx << 2) + (yy << 1) + zz;
}

struct BVHNode{
    int left = 0, right = 0;
    int objID = -1;
    aabb box;
};

__host__
bool buildBVHOnHost(std::vector<CudaObj> &objList, int start, int end){
    return false;
}


#endif //CUDARAYTRACER_BVH_H
