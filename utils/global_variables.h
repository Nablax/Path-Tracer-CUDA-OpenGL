//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_GLOBAL_VARIABLES_H
#define CUDARAYTRACER_GLOBAL_VARIABLES_H

namespace globalvar{
    const double kInfinity = std::numeric_limits<double>::infinity();
    const double kPi = 3.1415926535897932385;
    const double kSqrt2Div2 = 0.70710678;
    const double kSqrt3Div3 = 0.57735;
    const double kAspectRatio = 3.0 / 2.0;
    const double kDegToRad = 0.01745329252;
    const double kRadToDeg = 57.295779513;
    const int kFrameWidth = 1200;
    const int kFrameHeight = static_cast<int>(kFrameWidth / kAspectRatio);
    const int kSpp = 500;
    const int kMaxDepth = 50;
    const int kThreadX = 8;
    const int kThreadY = 8;
    const int kBlockX = kFrameWidth / kThreadX + 1;
    const int kBlockY = kFrameHeight / kThreadY + 1;
}

#endif //CUDARAYTRACER_GLOBAL_VARIABLES_H
