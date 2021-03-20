//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_GLOBAL_VARIABLES_H
#define CUDARAYTRACER_GLOBAL_VARIABLES_H

namespace globalvar{
//#define kInfinityGPU 99999999.0f
//#define kPiGPU 3.1415926f
//#define kPiGPUInv 0.318309f
//#define kDegToRadGPU 0.0174532f
    __device__
    const float kInfinityGPU = std::numeric_limits<float>::infinity();
    __device__
    const float kPiGPU = 3.1415926535897932385;
    __device__
    const float kPiGPUInv = 0.31830988618;
    __device__
    const float kDegToRadGPU = 0.01745329252;

    const float kSqrt2Div2 = 0.70710678;
    const float kSqrt3Div3 = 0.57735;
    const float kAspectRatio = 16.0 / 9.0;
    const float kInfinity = std::numeric_limits<float>::infinity();
    const float kDegToRad = 0.01745329252;
    const float kRadToDeg = 57.295779513;
    const int kFrameWidth = 800;
    const int kFrameHeight = static_cast<int>(kFrameWidth / kAspectRatio);
    const int kSpp = 1;
    const int kMaxDepth = 50;
    const int kThreadX = 8;
    const int kThreadY = 8;
    const int kBlockX = kFrameWidth / kThreadX + 1;
    const int kBlockY = kFrameHeight / kThreadY + 1;
    const float kCameraSpeed = 2.5f;
    const float kMouseSensitivity = 0.1f;
    __device__
    const float kRussianRoulette = 0.8f;
    __device__
    const float kInvRussianRoulette = 1.25f;
}

#endif //CUDARAYTRACER_GLOBAL_VARIABLES_H
