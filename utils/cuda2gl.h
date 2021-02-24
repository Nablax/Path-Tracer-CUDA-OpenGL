//
// Created by LZR on 2021/2/23.
//

#ifndef CUDARAYTRACER_CUDA2GL_H
#define CUDARAYTRACER_CUDA2GL_H
#include "OpenGL_Win/glad.c"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#include "cuda_check.h"

class Cuda2Gl{
public:
    Cuda2Gl(const int &fboCount, const int& width = globalvar::kFrameWidth, const int& height = globalvar::kFrameHeight){
        mBufferCount = fboCount;
        mCurIndex = 0;
        mFramebuffer = static_cast<GLuint *>(calloc(fboCount, sizeof(*(mFramebuffer))));
        mRenderbuffer = static_cast<GLuint *>(calloc(fboCount, sizeof(*(mRenderbuffer))));
        mCuGraphRes = static_cast<cudaGraphicsResource_t *>(calloc(fboCount, sizeof(*(mCuGraphRes))));
        mCuArray = static_cast<cudaArray_t *>(calloc(fboCount, sizeof(*(mCuArray))));

        glCreateRenderbuffers(fboCount, mRenderbuffer);
        glCreateFramebuffers(fboCount, mFramebuffer);
        for(int i = 0; i < fboCount; i++){
            glNamedFramebufferRenderbuffer(mFramebuffer[i], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mRenderbuffer[i]);
        }
        updateFrameSize(width, height);
    }
    void updateFrameSize(const int& width, const int& height){
        if(width == 0 || height == 0) return;
        mBufferWidth = width;
        mBufferHeight = height;
        for(int i = 0; i < mBufferCount; i++){
            if(mCuGraphRes[i] != nullptr)
                checkCudaErrors(cudaGraphicsUnregisterResource(mCuGraphRes[i]));
            glNamedRenderbufferStorage(mRenderbuffer[i], GL_RGBA8, width, height);

            cudaGraphicsGLRegisterImage(&mCuGraphRes[i],
                                        mRenderbuffer[i],
                                        GL_RENDERBUFFER,
                                        cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                        cudaGraphicsRegisterFlagsWriteDiscard);
        }
        checkCudaErrors(cudaGraphicsMapResources(mBufferCount, mCuGraphRes, 0));
        for(int i = 0; i < mBufferCount; i++){
            checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
                    &mCuArray[i],
                    mCuGraphRes[i], 0, 0));
        }
        checkCudaErrors(cudaGraphicsUnmapResources(mBufferCount, mCuGraphRes, 0));
    }
    void getFrameSize(int *const width, int *const height) const{
        *width = mBufferWidth;
        *height = mBufferHeight;
    }
    void mapGraphicResource(cudaStream_t stream){
        checkCudaErrors(cudaGraphicsMapResources(1, &mCuGraphRes[mCurIndex], stream));
    }
    void unMapGraphicResource(cudaStream_t stream){
        checkCudaErrors(cudaGraphicsUnmapResources(1, &mCuGraphRes[mCurIndex], stream));
    }
    cudaArray_const_t getCudaArray(){
        return mCuArray[mCurIndex];
    }
    int getCurBufferIdx() const{
        return mCurIndex;
    }
    void swapBuffer(){
        mCurIndex = (mCurIndex + 1) % mBufferCount;
    }
    void clearFramebuffer(){
        GLfloat color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glClearNamedFramebufferfv(mFramebuffer[mCurIndex], GL_COLOR, 0, color);
    }
    void blitFramebuffer(){
        glBlitNamedFramebuffer(mFramebuffer[mCurIndex], 0, 0, 0,
                          mBufferWidth, mBufferHeight, 0, 0,
                          mBufferWidth, mBufferHeight,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
    ~Cuda2Gl(){
        for(int i = 0; i < mBufferCount; i++){
            if(mCuGraphRes[i] != nullptr)
                cudaGraphicsUnregisterResource(mCuGraphRes[i]);
        }
        glDeleteRenderbuffers(mBufferCount, mRenderbuffer);
        glDeleteFramebuffers(mBufferCount, mFramebuffer);

        free(mFramebuffer);
        free(mRenderbuffer);
        free(mCuGraphRes);
        free(mCuArray);
    }
private:
    int mBufferCount = 2;
    int mCurIndex = 0;
    int mBufferWidth, mBufferHeight;
    GLuint* mFramebuffer;
    GLuint* mRenderbuffer;
    cudaGraphicsResource_t* mCuGraphRes;
    cudaArray_t* mCuArray;
};
#endif //CUDARAYTRACER_CUDA2GL_H
