#include <png_image.h>
#include <chrono>
#include <iostream>
#include "cuda_check.h"
#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"
#include "curand_kernel.h"

__device__ color ray_color(const ray& r, hittable_list *world, int depth) {
    hit_record rec;
    //printf("in ray color\n");
    if (world->hit(r, 0, globalvar::kInfinityGPU, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    vec3 unit_direction = vectorgpu::normalize(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void generateWorld(hittable_list *world){
    world->initObj(2);
    world->objects[0] = new sphere(point3(0,0,-1), 0.5);
    world->objects[1] = new sphere(point3(0,-100.5,-1), 100);
}

__global__ void clearWorld(hittable_list *world){
    delete world;
}

__global__ void initRandom(curandState *randState, int maxWidth, int maxHeight, int seed){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    int curPixel = row * maxWidth + col;
    curand_init(seed, curPixel, 0, &randState[curPixel]);
}

__global__ void render(vec3 *frameBuffer, int maxWidth, int maxHeight, int spp, int maxDepth,
                       camera *myCamera,
                       hittable_list *world, curandState *randState){
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    int curPixel = row * maxWidth + col;
    float maxWidthInv = 1.0f / maxWidth, maxHeightInv = 1.0f / maxHeight;

    for(int i = 0; i < spp; i++){
        float u = (col + curand_uniform(&randState[curPixel])) * maxWidthInv;
        float v = (row + curand_uniform(&randState[curPixel])) * maxHeightInv;
        ray r = myCamera->get_ray(u, v);
        //printf("%f %f %f\n", u, v, myCamera->fl);
        frameBuffer[curPixel] += ray_color(r, world, maxDepth);
    }
    frameBuffer[curPixel] /= spp;
}


int main()
{
    PngImage png(globalvar::kFrameWidth, globalvar::kFrameHeight);

    size_t frameBufferSize = globalvar::kFrameHeight * globalvar::kFrameWidth * sizeof(vec3);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    hittable_list *world;
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hittable_list)));

    camera *devCamera, *hostCamera = new camera();
    checkCudaErrors(cudaMalloc((void **)&devCamera, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(devCamera, hostCamera, sizeof(camera), cudaMemcpyHostToDevice));

    generateWorld<<<1, 1>>>(world);

    dim3 blocks(kBlockX, kBlockY);
    dim3 threads(kThreadX, kThreadY);
    //printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);

    curandState *devStates;
    checkCudaErrors(cudaMalloc((void **)&devStates, frameBufferSize * sizeof(curandState)));
    srand(time(nullptr));
    int seed = rand();
    initRandom<<<blocks, threads>>>(devStates, kFrameWidth, kFrameHeight, seed);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::system_clock::now();
    render<<<blocks, threads>>>(frameBuffer, kFrameWidth, kFrameHeight, kSpp, kMaxDepth,
                                devCamera, world, devStates);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    auto duration = end - start;
    std::cout <<  "Time cost "
                   << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
                   << "\n";
    for(int row = 0; row < kFrameHeight; row++){
        for(int col = 0; col < kFrameWidth; col++){
            int curPixel = row * kFrameWidth + col;
            //std::cerr << frameBuffer[curPixel] << '\n';
            png.saveColor(frameBuffer[curPixel], kFrameHeight - row - 1, col);
        }
    }

    png.write("../output/8.png");
    clearWorld<<<1, 1>>>(world);
    checkCudaErrors(cudaFree(frameBuffer));
    checkCudaErrors(cudaFree(world));
    delete []frameBuffer;
    return 0;
}

