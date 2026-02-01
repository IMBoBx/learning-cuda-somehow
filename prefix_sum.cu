#include <cuda_runtime.h>

#include <cuda/atomic>
#include <cuda/cmath>
#include <iostream>

#include "cudaTimer.cuh"

__global__ void prefix_sum_inclusive_genuinely_stupid(int length, float* in,
                                                      float* out) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    cuda::atomic_ref<float, cuda::thread_scope_system> res(out[x + y]);

    bool working = x < length && y < length && (x + y < length);

    if (working) {
        res.fetch_add(in[x + y]);
    }
}

__global__ void prefix_sum_exclusive(int length, float* in, float* out,
                                     float* partial_sums) {
    int tid = threadIdx.x;
    int gid = tid + blockDim.x * blockIdx.x;

    extern __shared__ float sums[];

    if (gid < length) {
        sums[tid] = in[gid];
    } else {
        sums[tid] = 0.0f;
    }

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;

        if (tid >= offset) {
            val = sums[tid - offset];
        }
        __syncthreads();

        sums[tid] += val;
        __syncthreads();
    }

    float inclusive = sums[tid];
    float prev_val = 0.0f;
    if (tid > 0) {
        prev_val = sums[tid - 1];
    }
    __syncthreads();
    if (gid < length) {
        out[gid] = prev_val;
    }

    if (tid == blockDim.x - 1 || gid == length - 1) {
        partial_sums[blockIdx.x] = inclusive;
    }
}

__global__ void add_block_offsets(int length, float* data, float* offsets) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < length) {
        data[gid] += offsets[blockIdx.x];
    }
}

void scan_launcher_recursive(int length, float* in, float* out) {
    int threads = 256;
    int blocks = cuda::ceil_div(length, threads);

    float* d_block_sums;
    cudaMallocManaged(&d_block_sums, blocks * sizeof(float));

    prefix_sum_exclusive<<<blocks, threads, threads * sizeof(float)>>>(
        length, in, out, d_block_sums);

    if (blocks > 1) {
        float* d_block_offsets;
        cudaMallocManaged(&d_block_offsets, blocks * sizeof(float));

        scan_launcher_recursive(blocks, d_block_sums, d_block_offsets);

        add_block_offsets<<<blocks, threads>>>(length, out, d_block_offsets);

        cudaFree(d_block_offsets);
    }

    cudaFree(d_block_sums);
}

int main() {
    int length = 65000;
    float* in;
    float* out;

    cudaMallocManaged(&in, length * sizeof(float));
    cudaMallocManaged(&out, length * sizeof(float));

    for (int i = 0; i < length; i++) {
        in[i] = 1.0f;
        out[i] = 0.0f;
    }

    // dim3 threads(32, 32);
    // dim3 blocks(cuda::ceil_div(length, threads.x),
    //             cuda::ceil_div(length, threads.y));

    // prefix_sum_inclusive_genuinely_stupid<<<blocks, threads>>>(length, in,
    // out);

    CudaTimer t;
    scan_launcher_recursive(length, in, out);
    cudaDeviceSynchronize();
    std::cout << t.stop_ms() << std::endl;

    std::cout << out[12290] << " " << out[42000] << std::endl;

    cudaFree(out);
    cudaFree(in);
}