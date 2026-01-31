#pragma once
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <math.h>

#define INDX(row, col, ld) (((row) * (ld)) + (col))


template <typename T = float>
struct CudaMatrix {
    int length = 128;
    T* data = nullptr;
    size_t bytes = 0;
};

template <typename T = float>
inline CudaMatrix<T> make_matrix(int length = 128) {
    CudaMatrix<T> m;
    m.length = length;
    m.bytes = length * length * sizeof(T);
    cudaMallocManaged(&m.data, m.bytes);
    return m;
}

template <typename T = float>
inline void free_matrix(CudaMatrix<T>& m) {
    if (m.data) {
        cudaFree(m.data);
        m.data = nullptr;
    }
}

template <typename T = float>
__global__ void init_matrix(CudaMatrix<T> A, float offset = 0.0f) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < A.length && col < A.length) {
        A.data[INDX(row, col, A.length)] =
            sinf((float)row + offset) + cosf((float)col - offset);
    }
}

template <typename T = float>
__global__ void transpose(CudaMatrix<T> A, CudaMatrix<T> C) {
    if (C.length != A.length) return;

    const int THREADS_PER_BLOCK_X = 32;
    const int THREADS_PER_BLOCK_Y = 32;

    int tileRow = blockDim.y * blockIdx.y;
    int tileCol = blockDim.x * blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int myRow = tileRow + tx;
    int myCol = tileCol + ty;

    __shared__ float TILE[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y + 1];

    if (myRow < A.length && myCol < A.length)
        TILE[tx][ty] = A.data[INDX(tileRow + ty, tileCol + tx, A.length)];

    __syncthreads();

    if (myRow < A.length && myCol < A.length)
        C.data[INDX(tileCol + ty, tileRow + tx, A.length)] = TILE[ty][tx];
}

template <typename T = float>
__global__ void matmul(CudaMatrix<T> A,
                       CudaMatrix<T> B,
                       CudaMatrix<T> C) {
    if (B.length != A.length || C.length != A.length) return;

    const int TILE_WIDTH = 32;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = ty + blockDim.y * blockIdx.y;
    int j = tx + blockDim.x * blockIdx.x;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    float val = 0;

    for (int phase = 0;
         phase < cuda::ceil_div(A.length, TILE_WIDTH);
         phase++) {

        int aCol = phase * TILE_WIDTH + tx;
        int bRow = phase * TILE_WIDTH + ty;

        tileA[ty][tx] = (i < A.length && aCol < A.length)
                            ? A.data[INDX(i, aCol, A.length)]
                            : 0.0f;
        tileB[ty][tx] = (bRow < B.length && j < B.length)
                            ? B.data[INDX(bRow, j, B.length)]
                            : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            val += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (i < C.length && j < C.length)
        C.data[INDX(i, j, C.length)] = val;
}
