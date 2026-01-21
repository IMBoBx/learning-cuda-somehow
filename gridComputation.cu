#include <cuda_runtime.h>

#include <iostream>
#include <cuda/cmath>

__global__ void computeGrid(float** A, int rows, int cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < rows && y < cols) {
        A[x][y] = sinf(x) + cosf(y);
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

    float* h_data = (float*)malloc(rows * cols * sizeof(float));
    float** h_A = (float**)malloc(rows * sizeof(float*));

    float* d_data = nullptr;
    float** d_A = nullptr;
    
    cudaMalloc(&d_data, rows * cols * sizeof(float));
    cudaMalloc(&d_A, rows * sizeof(float*));
    
    for (int i = 0; i < rows; i++) {
        h_A[i] = d_data + i * cols;
    }
    cudaMemcpy(d_A, h_A, rows * sizeof(float*), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(16, 16);

    std::cout << "starting computation" << std::endl;
    computeGrid<<<blocks, threads>>>(d_A, rows, cols);
    cudaDeviceSynchronize();
    std::cout << "done" << std::endl;
    
    cudaMemcpy(h_data, d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    
    for (int i = 0; i < rows && rows * cols <= 100; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%-2.5f   ", h_data[i * cols + j]);
        }
        std::cout << std::endl;
    }


    cudaFree(d_A);
    cudaFree(d_data);
    free(h_A);
    free(h_data);
} 
