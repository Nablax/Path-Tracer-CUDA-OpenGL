cmake_minimum_required(VERSION 3.17)
project(CUDARayTracer CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDARayTracer main.cu utils/png_image.h
        utils/utility.h utils/vec3.h utils/global_variables.h simulation/ray.h
        simulation/camera.h utils/cuda_check.h simulation/hittable.h
        simulation/sphere.h simulation/hittable_list.h simulation/material.h simulation/texture.h)

include_directories(outsource utils simulation)
set_target_properties(
        CUDARayTracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)