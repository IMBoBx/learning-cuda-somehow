#include <cuda_runtime.h>

#include <cuda/cmath>
#include <iostream>

__global__ void computeGrid(float* A, int rows, int cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < rows && y < cols) {
        A[x * cols + y] = sinf(x) + cosf(y);
    }
}

int main(int argc, char** argv) {
    int rows, cols;

    if (argc == 1) {
        rows = 5;
        cols = 5;
    } else if (argc == 2) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[1]);
    } else {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }

    int size = rows * cols * sizeof(float);

    std::cout << "starting memory ops" << std::endl;
    float* h_data = (float*)malloc(size);
    float* d_data = nullptr;
    cudaMalloc(&d_data, size);
    std::cout << "done" << std::endl;

    std::cout << "starting computation" << std::endl;

    dim3 threads(min(rows, 16), min(cols, 16));
    dim3 blocks(cuda::ceil_div(rows, threads.x),
                cuda::ceil_div(cols, threads.y));

    computeGrid<<<blocks, threads>>>(d_data, rows, cols);
    cudaDeviceSynchronize();

    std::cout << "done" << std::endl;

    std::cout << "starting memory ops" << std::endl;
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    std::cout << "done" << std::endl;

    for (int i = 0; i < rows && rows * cols <= 100; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%-2.5f   ", h_data[i * cols + j]);
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    free(h_data);
}
