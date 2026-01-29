#pragma once
#include <cuda_runtime.h>

struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    float stop_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};