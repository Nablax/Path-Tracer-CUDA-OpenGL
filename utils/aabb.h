//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_AABB_H
#define CUDARAYTRACER_AABB_H

class aabb {
public:
    __device__
    inline aabb() {}
    __device__
    inline aabb(const point3& a, const point3& b) { mMin = a; mMax = b;}

    __device__
    inline point3 getMin() const {return mMin; }
    __device__
    inline point3 getMax() const {return mMax; }

    __device__
    inline bool hit(const ray& r, float t_min, float t_max) const {
        vec3 rayOrig = r.origin(), rayDir = r.direction();
        for (int i = 0; i < 3; i++) {
            float dirInv = 1.0f / rayDir[i];
            float t0 = (mMin[i] - rayOrig[i]) * dirInv;
            float t1 = (mMin[i] - rayOrig[i]) * dirInv;
            if(dirInv < 0.0f) utils::swapGPU(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
    point3 mMin;
    point3 mMax;
};

namespace utils{
    __device__
    inline aabb unionBox(aabb box0, aabb box1){
        point3 newMin(fminf(box0.getMin().x(), box1.getMin().x()),
                      fminf(box0.getMin().y(), box1.getMin().y()),
                      fminf(box0.getMin().z(), box1.getMin().z()));

        point3 newMax(fmaxf(box0.getMax().x(), box1.getMax().x()),
                      fmaxf(box0.getMax().y(), box1.getMax().y()),
                      fmaxf(box0.getMax().z(), box1.getMax().z()));

        return {newMin, newMax};
    }
}


#endif //CUDARAYTRACER_AABB_H
