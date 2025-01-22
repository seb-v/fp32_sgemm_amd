
#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel2_lds.h"

#define TILE_SIZE 32

__global__ void kernel2_lds(const float *A, const float *B, float *C, int N)
{
    constexpr int BN = TILE_SIZE;
    constexpr int BK = TILE_SIZE;

    __shared__ float As[BN][BK];
    __shared__ float Bs[BK][BN];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N; t += BK)
    {
        Bs[threadIdx.y][threadIdx.x] = B[N * (threadIdx.y + t) + col];
        As[threadIdx.y][threadIdx.x] = A[N * row + t + threadIdx.x];

        __syncthreads();

        for (int k = 0; k < BK; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}
void Kernel2Lds::init()
{
}
void Kernel2Lds::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N)
{
    auto threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    auto blocksPerGrid = dim3(N / TILE_SIZE, N / TILE_SIZE);
    hipLaunchKernelGGL(kernel2_lds, blocksPerGrid, threadsPerBlock, 0, 0, d_a, d_b, d_c, N);
}

void Kernel2Lds::finalize()
{
}
