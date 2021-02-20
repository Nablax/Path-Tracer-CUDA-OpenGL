//
// Created by cosmos on 8/4/20.
//

#ifndef RAY_TRACING_PNG_IMAGE_H
#define RAY_TRACING_PNG_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "utility.h"

using namespace utils;
class PngImage {
public:
    PngImage(int w, int h, int n = 4): mDataW(w), mDataH(h), mDataN(n) {
        mImgSize = w * h * n;
        mData = new unsigned char[mImgSize];
    }
    PngImage(PngImage& other) = delete;
    PngImage& operator=(PngImage& other) = delete;

    void saveColor(color clr, int row, int col, int samples_per_pixel = 1){
        double crate = 1.0 / samples_per_pixel;
        getIdx(row, col, 0) = clamp((clr.x() * crate), 0, 0.999) * 256;
        getIdx(row, col, 1) = clamp((clr.y() * crate), 0, 0.999) * 256;
        getIdx(row, col, 2) = clamp((clr.z() * crate), 0, 0.999) * 256;
        getIdx(row, col, 3) = alpha_scale;
    }

    int width() { return mDataW; }
    int height() { return mDataH; }
    int channel() { return mDataN; }
    void write(const char* filename){
        stbi_write_png(filename, mDataW, mDataH, mDataN, mData, mDataW * 4);
    }
    ~PngImage() {
        delete [] mData;
    }
private:
    unsigned char* mData;
    int mDataW, mDataH, mDataN;
    int mImgSize;
    const double color_scale = 255.999, alpha_scale = 255.999;
    unsigned char& getIdx(int row, int col, int rgba){
        return *(mData + (row * mDataW + col) * mDataN + rgba);
    }
};
#endif //RAY_TRACING_PNG_IMAGE_H
