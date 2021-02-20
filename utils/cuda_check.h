//
// Created by LZR on 2021/2/19.
//

#ifndef CUDARAYTRACER_CUDA_CHECK_H
#define CUDARAYTRACER_CUDA_CHECK_H

#define checkCudaErrors(val) cudaCheck((val), #val, __FILE__, __LINE__)

void cudaCheck(cudaError_t result, const char *func, const char *file, const int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << ".\n" <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#endif //CUDARAYTRACER_CUDA_CHECK_H
