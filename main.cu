#include <png_image.h>
#include "cuda_check.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"

__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;
    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return -1.0;
    return (-half_b - sqrt(discriminant)) / a;
}

__device__ color ray_color(const ray& r) {
    float t = hit_sphere(point3(0,0,-1), 0.5, r);
    if (t > 0.0) {
        vec3 N = vectorgpu::normalize(r.at(t) - vec3(0,0,-1));
        return 0.5*color(N.x()+1, N.y()+1, N.z()+1);
    }
    vec3 unit_direction = vectorgpu::normalize(r.direction());
    t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}


__global__ void render(vec3 *frameBuffer, int maxWidth, int maxHeight,
                       vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical){
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    int curPixel = row * maxWidth + col;
    ray r(origin, lower_left_corner + col * horizontal / maxWidth + row * vertical / maxHeight - origin);
    frameBuffer[curPixel] = ray_color(r);
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

    size_t frameBufferSize = globalvar::kFrameHeight * globalvar::kFrameWidth * sizeof(vec3);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));

    dim3 blocks(kBlockX, kBlockY);
    dim3 threads(kThreadX, kThreadY);
    //printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    render<<<blocks, threads>>>(frameBuffer, kFrameWidth, kFrameHeight, origin, lower_left_corner, horizontal, vertical);
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
    checkCudaErrors(cudaFree(frameBuffer));
    delete []frameBuffer;
    return 0;
}

