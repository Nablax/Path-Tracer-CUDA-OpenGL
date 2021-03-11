//
// Created by LZR on 2021/3/10.
//

#ifndef CUDARAYTRACER_MORTON_CODE_H
#define CUDARAYTRACER_MORTON_CODE_H

#include "macros.h"

namespace morton{
    union Morton{
        unsigned long long mortonCode64;
        struct {
            unsigned long long objectID: 32;
            unsigned long long mortonCode: 32;
        };
    };

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
    unsigned int mortonCode3D(const vec3 &boxCenter, const aabb &maxBox)
    {
        point3 maxRange = maxBox.mMax - maxBox.mMin;
        float x = 0, y = 0, z = 0;
        if(maxRange.x() > 1e-7) x = (boxCenter.x() - maxBox.mMin.x()) / maxRange.x();
        if(maxRange.y() > 1e-7) y = (boxCenter.y() - maxBox.mMin.y()) / maxRange.y();
        if(maxRange.z() > 1e-7) z = (boxCenter.z() - maxBox.mMin.z()) / maxRange.z();
#ifdef TEST
        printf("x: %f, y: %f, z: %f\n", x, y, z);
#endif

        x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
        y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
        z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
        unsigned int xx = expandBits((unsigned int)x);
        unsigned int yy = expandBits((unsigned int)y);
        unsigned int zz = expandBits((unsigned int)z);
        return (xx << 2) + (yy << 1) + zz;
    }

    __device__ int clzMorton(Morton *sortedMortonUnion, int idx1, int idx2, const int &numObjects){
        if(idx1 < 0 || idx1 >= numObjects || idx2 < 0 || idx2 >= numObjects) return -1;
#ifdef MORTON32
        return __clz(sortedMortonUnion[idx1].mortonCode ^ sortedMortonUnion[idx2].mortonCode);
#else
        return __clzll(sortedMortonUnion[idx1].mortonCode64 ^ sortedMortonUnion[idx2].mortonCode64);
#endif
    }

    __device__ int clzMorton(unsigned long long code1, unsigned long long code2){
#ifdef MORTON32
        return __clz((code1 >> 32) ^ (code2 >> 32));
#else
        return __clzll(code1 ^ code2);
#endif
    }

    __host__
    auto computeMortonOnHost(std::vector<CudaObj> &objList, aabb &maxBox)->std::vector<morton::Morton>{
        std::vector<morton::Morton> myMorton(objList.size());
        for(int i = 0; i < objList.size(); i++){
            myMorton[i].objectID = i;
            myMorton[i].mortonCode = morton::mortonCode3D(objList[i].mBoundingBox.getCenter(), maxBox);
        }
        std::stable_sort(myMorton.begin(), myMorton.end(), [](const morton::Morton &a, const morton::Morton &b){
            return a.mortonCode < b.mortonCode;
        });
#ifdef TEST
        for(int i = 0; i < objList.size(); i++){
            printf("%llx %d\n", myMorton[i].mortonCode64, myMorton[i].objectID);
        }
#endif
        return myMorton;
    }
}


#endif //CUDARAYTRACER_MORTON_CODE_H
