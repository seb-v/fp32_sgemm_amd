
#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel1_naive.h"

__global__ void kernel1_naive(const float *A, const float *B, float *C, int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float acc_c = 0.0f; 
        for (int k = 0; k < K; ++k)
        {
            acc_c += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc_c + beta * C[row * N + col];
    }
}
void Kernel1Naive::init()
{
}
void Kernel1Naive::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N)
{
    auto threadsPerBlock = dim3(16,16);
    auto blocksPerGrid = dim3(N / 16, N / 16);
    hipLaunchKernelGGL(kernel1_naive, blocksPerGrid, threadsPerBlock, 0, 0, d_a, d_b, d_c, N,N,N, alpha, beta);
}
void Kernel1Naive::finalize()
{
}
