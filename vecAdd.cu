#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length) {
    std::srand(std::time({}));
    for (int i = 0; i < length; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C, int length) {
    for (int i = 0; i < length; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length,
                              float epsilon = 0.00001) {
    for (int i = 0; i < length; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void unifiedMemAccess(int vectorLength) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisionResult = (float*)malloc(vectorLength * sizeof(float));

    cudaMallocManaged(&A, vectorLength * sizeof(float));
    cudaMallocManaged(&B, vectorLength * sizeof(float));
    cudaMallocManaged(&C, vectorLength * sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    serialVecAdd(A, B, comparisionResult, vectorLength);

    int threads = 256;  // threads per block
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    cudaDeviceSynchronize();

    if (vectorApproximatelyEqual(comparisionResult, C, vectorLength)) {
        std::cout << "Unified Memory - Results Match" << std::endl;
    } else {
        std::cout << "Unified Memory - Results Mismatch" << std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisionResult);
}

void explicitMemAccess(int vectorLength) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisionResult = (float*)malloc(vectorLength * sizeof(float));

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMallocHost(&A, vectorLength * sizeof(float));
    cudaMallocHost(&B, vectorLength * sizeof(float));
    cudaMallocHost(&C, vectorLength * sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    serialVecAdd(A, B, comparisionResult, vectorLength);

    cudaMalloc(&devA, vectorLength * sizeof(float));
    cudaMalloc(&devB, vectorLength * sizeof(float));
    cudaMalloc(&devC, vectorLength * sizeof(float));

    cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(devC, 0, vectorLength * sizeof(float));

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);

    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDeviceToHost);

    if (vectorApproximatelyEqual(comparisionResult, C, vectorLength)) {
        std::cout << "Explicit Memory - Results Match" << std::endl;
    } else {
        std::cout << "Explicit Memory - Results Mismatch" << std::endl;
    }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    free(comparisionResult);
}

int main(int argc, char** argv) {
    int vectorLength = 1024;
    if (argc >= 2) {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemAccess(vectorLength);
    explicitMemAccess(vectorLength);
    return 0;
}