cmake_minimum_required(VERSION 3.17)
project(CUDARayTracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDARayTracer main.cu OpenGL_Win/glad.c utils/cuda2gl.h)

include_directories(${PROJECT_SOURCE_DIR}/OpenGL_Win/include)
link_libraries(${PROJECT_SOURCE_DIR}/OpenGL_Win/lib)
target_link_libraries(CUDARayTracer -lopengl32 ${PROJECT_SOURCE_DIR}/OpenGL_Win/lib/glfw3.lib)

include_directories(outsource utils simulation)
set_target_properties(
        CUDARayTracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)