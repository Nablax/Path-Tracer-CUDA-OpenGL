#include <png_image.h>
#include "cuda_check.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"

__device__ color ray_color(const ray& r, hittable_list *world) {
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


__global__ void render(vec3 *frameBuffer, int maxWidth, int maxHeight,
                       vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, hittable_list *world){
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    int curPixel = row * maxWidth + col;
    ray r(origin, lower_left_corner + col * horizontal / maxWidth + row * vertical / maxHeight - origin);
    frameBuffer[curPixel] = ray_color(r, world);
}


int main()
{
    float viewport_height = 2.0;
    float viewport_width = globalvar::kAspectRatio * viewport_height;
    float focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    PngImage png(globalvar::kFrameWidth, globalvar::kFrameHeight);

    size_t frameBufferSize = globalvar::kFrameHeight * globalvar::kFrameWidth * sizeof(vec3);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    hittable_list *world;
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hittable_list)));

    generateWorld<<<1, 1>>>(world);

    dim3 blocks(kBlockX, kBlockY);
    dim3 threads(kThreadX, kThreadY);
    //printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    render<<<blocks, threads>>>(frameBuffer, kFrameWidth, kFrameHeight, origin, lower_left_corner, horizontal, vertical, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    for(int row = 0; row < kFrameHeight; row++){
        for(int col = 0; col < kFrameWidth; col++){
            int curPixel = row * kFrameWidth + col;
            //std::cerr << frameBuffer[curPixel] << '\n';
            png.saveColor(frameBuffer[curPixel], kFrameHeight - row - 1, col);
        }
    }

    png.write("../output/6.png");
    //clearWorld<<<1, 1>>>(world);
    checkCudaErrors(cudaFree(frameBuffer));
    checkCudaErrors(cudaFree(world));
    delete []frameBuffer;
    return 0;
}

