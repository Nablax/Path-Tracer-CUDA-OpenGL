//
// Created by LZR on 2021/2/22.
//

#ifndef CUDARAYTRACER_AABB_H
#define CUDARAYTRACER_AABB_H

class aabb {
public:
    __host__ __device__
    inline aabb() {}
    __host__ __device__
    inline aabb(const point3& a, const point3& b) { mMin = a; mMax = b;}

    __host__ __device__
    inline point3 getMin() const {return mMin; }
    __host__ __device__
    inline point3 getMax() const {return mMax; }

    __device__
    inline bool hit(const Ray& r, float t_min, float t_max) const {
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
    __host__ __device__
    inline void unionBoxInPlace(const aabb &box){
        mMin.e[0] = fminf(mMin.x(), box.getMin().x());
        mMin.e[1] = fminf(mMin.y(), box.getMin().y());
        mMin.e[2] = fminf(mMin.z(), box.getMin().z());

        mMax.e[0] = fmaxf(mMax.x(), box.getMax().x());
        mMax.e[1] = fmaxf(mMax.y(), box.getMax().y());
        mMax.e[2] = fmaxf(mMax.z(), box.getMax().z());
    }
    point3 mMin;
    point3 mMax;
};

namespace utils{
    __host__ __device__
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
