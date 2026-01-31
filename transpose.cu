#include <cuda_runtime.h>

#include <cuda/cmath>

#include "cudaTimer.cuh"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define INDX(row, col, ld) (((row) * (ld)) + (col))

__global__ void initMatrix(float* A, int length) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < length && col < length) {
        A[INDX(row, col, length)] = sinf(row) + cosf(col);
    }
}

__global__ void naiveTranspose(int m, float* a, float* c) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < m && col < m) {
        c[INDX(col, row, m)] = a[INDX(row, col, m)];
    }
}

// using shared memory
__global__ void optimisedTranspose(int m, float* A, float* C) {
    int tileRow = blockDim.y * blockIdx.y;
    int tileCol = blockDim.x * blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int myRow = tileRow + tx;
    int myCol = tileCol + ty;

    __shared__ float TILE[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y + 1];

    if (myRow < m && myCol < m)
        TILE[tx][ty] = A[INDX(tileRow + ty, tileCol + tx, m)];

    __syncthreads();

    if (myRow < m && myCol < m)
        C[INDX(tileCol + ty, tileRow + tx, m)] = TILE[ty][tx];
}

int main() {
    int m = 8192;
    int SIZE = m * m * sizeof(float);

    float* A = nullptr;
    float* C = nullptr;
    float* D = nullptr;

    cudaMallocManaged(&A, SIZE);
    cudaMallocManaged(&C, SIZE);
    cudaMallocManaged(&D, SIZE);

    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks(cuda::ceil_div(m, threads.x), cuda::ceil_div(m, threads.y));

    initMatrix<<<blocks, threads>>>(A, m);
    cudaDeviceSynchronize();

    CudaTimer t1;
    naiveTranspose<<<blocks, threads>>>(m, A, C);
    cudaDeviceSynchronize();
    printf("Naive time: %fms\n", t1.stop_ms());

    printf("%f %f\n", A[INDX(12, 500, m)], C[INDX(500, 12, m)]);
    printf("%f %f\n", A[INDX(24, 233, m)], C[INDX(233, 24, m)]);


    CudaTimer t2;
    optimisedTranspose<<<blocks, threads>>>(m, A, D);
    cudaDeviceSynchronize();
    printf("\nOptimised time: %fms\n", t2.stop_ms());

    printf("%f %f\n", A[INDX(12, 500, m)], D[INDX(500, 12, m)]);
    printf("%f %f\n", A[INDX(24, 233, m)], D[INDX(233, 24, m)]);

    cudaFree(D);
    cudaFree(C);
    cudaFree(A);
}