//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_BVH_H
#define CUDARAYTRACER_BVH_H

#include "cuda_object.h"
#include <vector>
#include <algorithm>
#include "morton_code.h"
#include "aabb.h"
#include "macros.h"
#include "bvh_node.h"

namespace lbvh{
    __device__
    inline int2 determineRange(morton::Morton* sortedMortonUnion, int numObjects, int idx){
        int leftDelta = clzMorton(sortedMortonUnion, idx, idx - 1, numObjects);
        int rightDelta = clzMorton(sortedMortonUnion, idx, idx + 1, numObjects);
        int gradientDirection = utils::sign(rightDelta - leftDelta);
        int minDelta = min(leftDelta, rightDelta);

        int maxStride = 2;
        while(clzMorton(sortedMortonUnion,
                        idx, idx + maxStride * gradientDirection,
                        numObjects) > minDelta){
            maxStride *= 2;
        }
        int l = 0;
        for(int curStride = (maxStride >> 1); curStride >= 1; (curStride >>= 1)){
            if(clzMorton(sortedMortonUnion, idx, idx + (l + curStride) * gradientDirection, numObjects) > minDelta){
                l += curStride;
            }
        }
        int jdx = idx + l * gradientDirection;
        if(gradientDirection < 0) utils::swapGPU(jdx, idx);
        return make_int2(idx, jdx);

    }

    __device__
    inline int findSplit(morton::Morton* sortedMortonUnion, int first, int last){
#ifdef MORTON32
        unsigned int firstCode = sortedMortonUnion[first].mortonCode64;
        unsigned int lastCode = sortedMortonUnion[last].mortonCode64;
#else
        unsigned long long firstCode = sortedMortonUnion[first].mortonCode64;
        unsigned long long lastCode = sortedMortonUnion[last].mortonCode64;
#endif
        if(first == last) return (first + last) >> 1;
        int commonPrefix = morton::clzMorton(firstCode , lastCode);

        int split = first, step = last - first;
        do{
            step = (step + 1) >> 1;
            int newSplit = split + step;
            if(newSplit < last){
#ifdef MORTON32
                unsigned int splitCode = sortedMortonUnion[newSplit].mortonCode;
#else
                unsigned long long splitCode = sortedMortonUnion[newSplit].mortonCode64;
#endif
                int splitPrefix = morton::clzMorton(firstCode, splitCode);
                if(splitPrefix > commonPrefix) split = newSplit;
            }
        } while (step > 1);
        return split;
    }

    __global__
    void generateLBVH(morton::Morton *sortedMortonUnion, BVHNode* nodes, CudaObj* obj, int numObjects){
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

#ifdef TEST
            printf("node: %d, range: (%d, %d), split: %d, childA: %d, childB: %d\n", idx, first, last, split, childA, childB);
#endif
            nodes[idx].left = childA;
            nodes[idx].right = childB;

            __syncthreads();

            nodes[childA].parent = idx;
            nodes[childB].parent = idx;
        }
    }

    __global__
    void growBBox(BVHNode* nodes, int numObjects){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx >= numObjects) return;
        int leafNodeStartIdx = numObjects - 1;
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
    bool buildBVH(std::vector<CudaObj> &objList, CudaObj *objListOnDevice, aabb& maxBox, BVHNode* &lbvhArrayDevice){
        auto myMorton = morton::computeMortonOnHost(objList, maxBox);
        int numObj = objList.size();
        checkCudaErrors(cudaMalloc((void **)&lbvhArrayDevice, (2 * numObj - 1) * sizeof(BVHNode)));
        morton::Morton* sortedMortonUnion;
        checkCudaErrors(cudaMalloc((void **)&sortedMortonUnion, myMorton.size() * sizeof(morton::Morton)));
        checkCudaErrors(cudaMemcpy(sortedMortonUnion, myMorton.data(), myMorton.size() * sizeof(morton::Morton), cudaMemcpyHostToDevice));
        generateLBVH<<<static_cast<int>(numObj / 64) + 1, 64>>>(sortedMortonUnion, lbvhArrayDevice, objListOnDevice, numObj);
        checkCudaErrors(cudaDeviceSynchronize());
        growBBox<<<static_cast<int>(numObj / 64) + 1, 64>>>(lbvhArrayDevice, numObj);
        checkCudaErrors(cudaDeviceSynchronize());
        return false;
    }
}


#endif //CUDARAYTRACER_BVH_H
