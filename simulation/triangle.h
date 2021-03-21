//
// Created by LZR on 2021/3/14.
//

#ifndef CUDARAYTRACER_TRIANGLE_H
#define CUDARAYTRACER_TRIANGLE_H

class Triangle{
public:
    __host__ __device__
    Triangle(){};
    __host__ __device__
    Triangle(point3 v0, point3 v1, point3 v2){
        mVertex[0] = v0;
        mVertex[1] = v1;
        mVertex[2] = v2;
        mNormal = vectorgpu::normalize(
                vectorgpu::cross(mVertex[1] - mVertex[0], mVertex[2] - mVertex[0])
                );
    }
    point3 mVertex[3];
    point3 mNormal;
};

#endif //CUDARAYTRACER_TRIANGLE_H
