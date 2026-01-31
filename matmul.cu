#include <cuda_runtime.h>

#include <cuda/cmath>
#include <iostream>

#include "cudaTimer.cuh"

#define TILE_WIDTH 32

#define INDX(row, col, ld) ((row * ld) + col)

__global__ void initMatrix(float* A, int length, float offset) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < length && col < length) {
        A[INDX(row, col, length)] =
            sinf((float)row + offset) + cosf((float)col - offset);
    }
}

__global__ void naiveMatmul(int m, float* A, float* B, float* C) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= m || col >= m) return;

    float sum = 0;
    for (int i = 0; i < m; i++) {
        sum += A[INDX(row, i, m)] * B[INDX(i, col, m)];
    }
    C[INDX(row, col, m)] = sum;
}

__global__ void naiveMatmulButBetter(int m, float* A, float* B, float* C) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= m || col >= m) return;

    float sum = 0;
    for (int i = 0; i < m; i++) {
        sum += A[INDX(row, i, m)] * B[INDX(i, col, m)];
    }
    C[INDX(row, col, m)] = sum;
}

__global__ void tiledMatmul(int m, float* A, float* B, float* C) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = ty + blockDim.y * blockIdx.y;
    int j = tx + blockDim.x * blockIdx.x;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    float val = 0;

    for (int phase = 0; phase < cuda::ceil_div(m, TILE_WIDTH); phase++) {
        int aCol = phase * TILE_WIDTH + tx;
        int bRow = phase * TILE_WIDTH + ty;

        tileA[ty][tx] = (i < m && aCol < m) ? A[INDX(i, aCol, m)] : 0.0f;
        tileB[ty][tx] = (bRow < m && j < m) ? B[INDX(bRow, j, m)] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            val += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (i < m && j < m) C[INDX(i, j, m)] = val;
}

int main() {
    int m = 1027;
    int SIZE = m * m * sizeof(float);

    float* A = nullptr;
    float* B = nullptr;
    float* C1 = nullptr;  // naive
    float* C2 = nullptr;  // naive but better
    float* C3 = nullptr;  // tiled

    cudaMallocManaged(&A, SIZE);
    cudaMallocManaged(&B, SIZE);
    cudaMallocManaged(&C1, SIZE);
    cudaMallocManaged(&C2, SIZE);
    cudaMallocManaged(&C3, SIZE);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(cuda::ceil_div(m, threads.x), cuda::ceil_div(m, threads.y));

    initMatrix<<<blocks, threads>>>(A, m, 0.1f);
    initMatrix<<<blocks, threads>>>(B, m, 0.2f);

    CudaTimer t1;  // naive
    naiveMatmul<<<blocks, threads>>>(m, A, B, C1);
    cudaDeviceSynchronize();
    std::cout << "Naive: " << t1.stop_ms() << std::endl;

    CudaTimer t2;  // naive but better
    naiveMatmulButBetter<<<blocks, threads>>>(m, A, B, C2);
    cudaDeviceSynchronize();
    std::cout << "Naive but better: " << t2.stop_ms() << std::endl;

    CudaTimer t3;  // tiled
    tiledMatmul<<<blocks, threads>>>(m, A, B, C3);
    cudaDeviceSynchronize();
    std::cout << "Tiled: " << t3.stop_ms() << std::endl;

    std::cout << std::endl
              << C1[INDX(30, 102, m)] << " " << C2[INDX(30, 102, m)] << " "
              << C3[INDX(30, 102, m)] << std::endl;

    cudaFree(C3);
    cudaFree(C2);
    cudaFree(C1);
    cudaFree(B);
    cudaFree(A);
}