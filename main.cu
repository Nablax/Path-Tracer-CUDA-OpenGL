#include <png_image.h>
#include "cuda_check.h"
#include "ray.h"

__global__ void render(float *frameBuffer, int maxWidth, int maxHeight){
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    int curPixel = (row * maxWidth + col) * 3;
    frameBuffer[curPixel] = (float)row / maxHeight;
    frameBuffer[curPixel + 1] = (float)col / maxWidth;
    frameBuffer[curPixel + 2] = 0.25;
}

__device__ color ray_color(const ray &r){
    vec3 unit_direction = vectorgpu::normalize(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main()
{
    auto viewport_height = 2.0;
    auto viewport_width = globalvar::kAspectRatio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    PngImage png(globalvar::kFrameWidth, globalvar::kFrameHeight);

    size_t frameBufferSize = 3 * globalvar::kFrameHeight * globalvar::kFrameWidth * sizeof(float);
    float *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    dim3 blocks(kBlockX, kBlockY);
    dim3 threads(kThreadX, kThreadY);
    //printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    render<<<blocks, threads>>>(frameBuffer, kFrameWidth, kFrameHeight);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    for(int row = 0; row < kFrameHeight; row++){
        for(int col = 0; col < kFrameWidth; col++){
            int curPixel = (row * kFrameWidth + col) * 3;
            //std::cerr << frameBuffer[curPixel] << '\n';
            png.saveColor(color(frameBuffer[curPixel], frameBuffer[curPixel + 1], frameBuffer[curPixel + 2]), row, col);
        }
    }

    png.write("../output/output.png");
    checkCudaErrors(cudaFree(frameBuffer));
    delete []frameBuffer;
    return 0;
}

