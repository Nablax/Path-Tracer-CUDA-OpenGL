//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_BVH_H
#define CUDARAYTRACER_BVH_H

#include "cuda_object.h"
#include <vector>
#include <algorithm>
#include "aabb.h"

struct BVHNode{
    int left = -1, right = -1;
    int parent = -1;
    int objID = -1;
    aabb box;
};

union Morton{
    unsigned long long mortonCode64;
    struct {
        unsigned long long objectID: 32;
        unsigned long long mortonCode: 32;
    };
};

extern BVHNode* bvhArrayDevice;

__host__ __device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__host__ __device__
unsigned int mortonCode3D(vec3 boxCenter)
{
    float x = boxCenter.x();
    float y = boxCenter.y();
    float z = boxCenter.z();
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx << 2) + (yy << 1) + zz;
}

__device__
inline int2 determineRange(Morton* sortedMortonUnion, int numObjects, int idx){
    if(idx == 0) return make_int2(0, numObjects - 1);
    unsigned int currentMortonCode = sortedMortonUnion[idx].mortonCode;
    int leftDelta = __clz(currentMortonCode ^ sortedMortonUnion[idx - 1].mortonCode);
    int rightDelta = __clz(currentMortonCode ^ sortedMortonUnion[idx + 1].mortonCode);
    int gradientDirection = (rightDelta > leftDelta) ? 1: -1;
    int minDelta = min(leftDelta, rightDelta);

    int maxStride = 2, tmpDelta = -1;
    int tmpIdx = idx + gradientDirection * maxStride;

    if(tmpIdx >= 0 && tmpIdx < numObjects){
        tmpDelta = __clz(currentMortonCode ^ sortedMortonUnion[tmpIdx].mortonCode);
    }

    while(tmpDelta > minDelta){
        maxStride <<= 1;
        tmpIdx = idx + gradientDirection * maxStride;
        tmpDelta = -1;

        if(tmpIdx >= 0 && tmpIdx < numObjects){
            tmpDelta = __clz(currentMortonCode ^ sortedMortonUnion[tmpIdx].mortonCode);
        }
    }

    int l = 0;
    int curStride = maxStride >> 1;
    while(curStride > 0){
        tmpIdx = idx + (l + curStride) * gradientDirection;
        tmpDelta = -1;
        if(tmpIdx >= 0 && tmpIdx < numObjects){
            tmpDelta = __clz(currentMortonCode ^ sortedMortonUnion[tmpIdx].mortonCode);
        }
        if(tmpDelta > minDelta){
            l += curStride;
        }
        curStride >>= 1;
    }
    int jdx = idx + l * curStride;
    if(gradientDirection < 0) utils::swapGPU(jdx, idx);
    return make_int2(idx, jdx);
}

__device__
inline int findSplit(Morton* sortedMortonUnion, int first, int last){
    unsigned int firstCode = sortedMortonUnion[first].mortonCode;
    unsigned int lastCode = sortedMortonUnion[last].mortonCode;
    if(first == last) return (first + last) >> 1;
    int commonPrefix = __clz((firstCode ^ lastCode));

    int split = first, step = last - first;
    do{
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if(newSplit < last){
            unsigned int splitCode = sortedMortonUnion[newSplit].mortonCode;
            int splitPrefix = __clz((firstCode ^ splitCode));
            if(splitPrefix > commonPrefix) split = newSplit;
        }
    } while (step > 1);
    return split;
}

__global__
void generateLBVH(Morton *sortedMortonUnion, BVHNode* nodes, CudaObj* obj, int numObjects){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numObjects) return;

    int leafNodeStartIdx = numObjects - 1;
    int curObjID = sortedMortonUnion[idx].objectID;
    BVHNode curLeafNode;
    curLeafNode.objID = curObjID;
    curLeafNode.box = obj[curObjID].mBoundingBox;
    nodes[leafNodeStartIdx + idx] = curLeafNode;
    if(idx < numObjects - 1){
        BVHNode curInternalNode;
        nodes[idx] = curInternalNode;
    }

    __syncthreads();

    if(idx < numObjects - 1) {

        int2 range = determineRange(sortedMortonUnion, numObjects, idx);
        int first = range.x;
        int last = range.y;

        int split = findSplit(sortedMortonUnion, first, last);

        int childA, childB;
        if(split == first) childA = leafNodeStartIdx + split;
        else childA = split;

        if(split + 1 == last) childB = leafNodeStartIdx + split + 1;
        else childB = split + 1;

        nodes[idx].left = childA;
        nodes[idx].right = childB;
        nodes[childA].parent = idx;
        nodes[childB].parent = idx;
    }

    __syncthreads();

    int parentNodeIdx = nodes[leafNodeStartIdx + idx].parent;
    while(parentNodeIdx >= 0){
        BVHNode parentNode = nodes[parentNodeIdx];
        aabb childUnionBox = unionBox(nodes[parentNode.left].box, nodes[parentNode.right].box);
        parentNode.box.unionBoxInPlace(childUnionBox);
        nodes[parentNodeIdx] = parentNode;
        parentNodeIdx = parentNode.parent;
    }
}

__host__
auto computeMortonOnHost(std::vector<CudaObj> &objList)->std::vector<Morton>{
    std::vector<Morton> myMorton(objList.size());
    for(int i = 0; i < objList.size(); i++){
        myMorton[i].objectID = i;
        myMorton[i].mortonCode = mortonCode3D(objList[i].mBoundingBox.getCenter());
    }
    std::stable_sort(myMorton.begin(), myMorton.end(), [](const Morton &a, const Morton &b){
       return a.mortonCode < b.mortonCode;
    });
    return myMorton;
}
__host__
bool buildBVH(std::vector<CudaObj> &objList, CudaObj *objListOnDevice){
    auto myMorton = computeMortonOnHost(objList);
    checkCudaErrors(cudaFree(bvhArrayDevice));
    int numObj = objList.size();
    checkCudaErrors(cudaMalloc((void **)&bvhArrayDevice, (2 * numObj - 1) * sizeof(BVHNode)));
    Morton* sortedMortonUnion;
    checkCudaErrors(cudaMalloc((void **)&sortedMortonUnion, myMorton.size() * sizeof(Morton)));
    checkCudaErrors(cudaMemcpy(sortedMortonUnion, myMorton.data(), myMorton.size() * sizeof(Morton), cudaMemcpyHostToDevice));
    generateLBVH<<<static_cast<int>(numObj / 64) + 1, 64>>>(sortedMortonUnion, bvhArrayDevice, objListOnDevice, numObj);
    return false;
}


#endif //CUDARAYTRACER_BVH_H
