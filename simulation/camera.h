//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_CAMERA_H
#define RAY_TRACING_CAMERA_H

#include "ray.h"

class camera {
public:
    __host__ __device__ camera(
        point3 lookFrom,
        point3 lookAt,
        float vfov, // vertical field-of-view in degrees
        float aspect_ratio,
        float aperture,
        float focus_dist,
        float _time0 = 0,
        float _time1 = 0
    ) {
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta/2);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        mFront = vectorgpu::normalize(lookFrom - lookAt);
        mRight = vectorgpu::normalize(cross(vec3(0, 1, 0), mFront));
        mUp = cross(mFront, mRight);

        mPosition = lookFrom;
        mHorizontalViewportSize = focus_dist * viewport_width * mRight;
        mVerticalViewportSize = focus_dist * viewport_height * mUp;
        mViewportLowLeftCorner = mPosition - mHorizontalViewportSize/2 - mVerticalViewportSize/2 - focus_dist * mFront;
        mLensRadius = aperture / 2;
        time0 = _time0;
        time1 = _time1;
        mFocusDist = focus_dist;
    }

    __host__ void processKeyboard(utils::directions dir, float deltaTime){
        float velocity = globalvar::kCameraSpeed * deltaTime;
        if (dir == FORWARD)
            mPosition += mFront * velocity;
        if (dir == BACKWARD)
            mPosition -= mFront * velocity;
        if (dir == LEFT)
            mPosition -= mRight * velocity;
        if (dir == RIGHT)
            mPosition += mRight * velocity;
        mViewportLowLeftCorner = mPosition - mHorizontalViewportSize/2 - mVerticalViewportSize/2 - mFocusDist * mFront;
    }

    __device__ inline ray get_ray(float s, float t, curandState *randState) const {
        vec3 rd = mLensRadius * utils::randomInUnitDisk(randState);
        vec3 offset = mRight * rd.x() + mUp * rd.y();
        return {mPosition + offset,
                mViewportLowLeftCorner + s * mHorizontalViewportSize + t * mVerticalViewportSize - mPosition - offset,
                utils::randomInRange(time0, time1, randState)};
    }

public:
    point3 mPosition;
    point3 mViewportLowLeftCorner;
    vec3 mHorizontalViewportSize;
    vec3 mVerticalViewportSize;
    vec3 mRight, mUp, mFront;
    float mFocusDist;
    float mLensRadius;
    float time0, time1;
    float mYaw, mPitch;
};

#endif //RAY_TRACING_CAMERA_H
