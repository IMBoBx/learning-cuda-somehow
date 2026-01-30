#include <cuda_runtime.h>

#include <cuda/atomic>
#include <iostream>

__global__ void broken_reduce(float* vector, float* res, int length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (workIndex < length) {
        res[0] += vector[workIndex];
    }
}

__global__ void sumReduce(float* vector, float* res, int length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    cuda::atomic_ref<float, cuda::thread_scope_device> sum(*res);
    if (workIndex < length) {
        sum.fetch_add(vector[workIndex], cuda::memory_order_relaxed);
    }
}

int main() {
    int length = 1024;
    float* vector;
    float* res;

    cudaMallocManaged(&vector, length * sizeof(float));
    cudaMallocManaged(&res, sizeof(float));

    for (int i = 0; i < length; i++) {
        vector[i] = 1.0f;
    }

    res[0] = 0.0f;

    sumReduce<<<32, 128>>>(vector, res, length);
    cudaDeviceSynchronize();

    std::cout << res[0] << std::endl;

    cudaFree(vector);
    cudaFree(res);
}
