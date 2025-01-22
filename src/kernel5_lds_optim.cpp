#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel5_lds_optim.h"
  
#define BLOCK_SIZE 128
template<int UNROLL>
__global__ void 
__launch_bounds__(BLOCK_SIZE) 
kernel5_lds_optim(float *a, float *b, float *c, int N, float alpha, float beta)
{
    // Block Tile size (square)
    constexpr int BN = 128;
    constexpr int BM = 128;
    constexpr int BK = 8; // Number of Row or column we read per K blocks

    // Thread Tile size . 4x4 for LDS 128 reads
    constexpr int TN = 4;
    constexpr int TM = 4;

    constexpr int nbWarps = BLOCK_SIZE / 32;
    // Warp Tile size : 128x32
    constexpr int WN = 128;
    constexpr int WM = BN * BM / nbWarps / WN;

    // Number of warp on X & Y axis in the Block tile
    constexpr int nbWarpX = BN / WN;
    constexpr int nbWarpY = BM / WM;

    const int warpIndex = threadIdx.x / 32;
    const int warpIdx = warpIndex % nbWarpX;
    const int warpIdy = warpIndex / nbWarpX;
    const int indexInWarp = threadIdx.x % 32;

    // A warp is a block of 8x4 of the output matrix
    constexpr int nbThreadXPerWarp = 8;
    constexpr int nbThreadYPerWarp = 4;

    // Thread coordinates in Warp
    const int idxInWarp = indexInWarp % nbThreadXPerWarp;
    const int idyInWarp = indexInWarp / nbThreadXPerWarp;

    constexpr int nbIterWarpN = WN / (nbThreadXPerWarp * TN);
    constexpr int nbIterWarpM = WM / (nbThreadYPerWarp * TM);

    // Warp Sub-tile size
    constexpr int SUBWN = WN / nbIterWarpN;
    constexpr int SUBWM = WM / nbIterWarpM;

    // Thread mapping to read BKxBN block from A
    int rAIdx = threadIdx.x % BK;
    int rAIdy = threadIdx.x / BK;
    // Thread mapping to read BNxBK block from B
    int rBIdx = threadIdx.x % BN;
    int rBIdy = threadIdx.x / BN;

    constexpr int strideReadB = BLOCK_SIZE / BN;
    constexpr int strideReadA = BLOCK_SIZE / BK;
    constexpr int nbReadsB = BN * BK / BLOCK_SIZE;
    constexpr int nbReadsA = BM * BK / BLOCK_SIZE;

    float A_col[nbIterWarpM * TM];
    float B_row[nbIterWarpN * TN];

    __shared__ float As[BK][BM+4]; // 4 padding to avoid bank conflicts
    __shared__ float Bs[BK][BN];

    float c_regs[TM * nbIterWarpM * TN * nbIterWarpN] = {0.0f};

    for (int i = 0; i < nbReadsB; i++)
    {
        int index_x = BN * blockIdx.x + rBIdx;
        int index_y = rBIdy + i * strideReadB;
        Bs[index_y % BK][index_x % BN] = b[N * index_y + index_x];
    }
    for (int i = 0; i < nbReadsA; i++)
    {
        int index_x = rAIdx;
        int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
        As[(index_x % BK)][(index_y % BM)] = a[N * index_y + index_x];
    }

        __syncthreads();
    // Iteration over BK blocks.
    for (int kId = 0; kId < N; kId += BK)
    {
        float regA[nbReadsA];
        float regB[nbReadsB];
        if (kId < N - 1)
        {
            // We populate the Shared Memory with Ks row and columns
            for (int i = 0; i < nbReadsB; i++)
            {
                int index_x = BN * blockIdx.x + rBIdx;
                int index_y = rBIdy + i * strideReadB + kId + BK;
                regB[i] = b[N * index_y + index_x];
            }

            for (int i = 0; i < nbReadsA; i++)
            {
                int index_x = rAIdx + kId + BK;
                int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
                regA[i] = a[N * index_y + index_x]; 
            }
        }


#pragma unroll UNROLL
        for (int k = 0; k < BK; k += 1)
        {
            // we cache A & B for the entire Warp tile
            for (int iterWarp = 0; iterWarp < nbIterWarpN; iterWarp++)
            {
                for (int i = 0; i < TN; i++)
                {
                    int index = warpIdx * WN +     // warpId
                                iterWarp * SUBWN + // warp subtile
                                TN * idxInWarp +
                                +i;
                    B_row[iterWarp * TN + i] = Bs[k][index];
                }
            }

            for (int iterWarp = 0; iterWarp < nbIterWarpM; iterWarp++)
            {
                for (int i = 0; i < TM; i++)
                {
                    int index = warpIdy * WM +     // warpId
                                iterWarp * SUBWM + // warp subtile
                                TM * idyInWarp +
                                i;

                    A_col[iterWarp * TM + i] = As[k][index]; // TMP
                }
            }

            // we accumulate to C_regs
            for (int iterWarpM = 0; iterWarpM < nbIterWarpM; iterWarpM++)
            {
                for (int iterWarpN = 0; iterWarpN < nbIterWarpN; iterWarpN++)
                {
                    for (int yt = 0; yt < TM; yt++)
                    {
                        for (int xt = 0; xt < TN; xt++)
                        {
                            const int x = iterWarpN * TN + xt;
                            const int y = iterWarpM * TM + yt;
                            c_regs[y * TN * nbIterWarpN + x] += A_col[y] * B_row[x];
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (kId < N - 1)
        {
            for (int i = 0; i < nbReadsB; i++)
            {
                int index_x = BN * blockIdx.x + rBIdx;
                int index_y = rBIdy + i * strideReadB + kId + BK;
                Bs[index_y % BK][index_x % BN] = regB[i]; // row
            }

            for (int i = 0; i < nbReadsA; i++)
            {
                int index_x = rAIdx + kId + BK;
                int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
                As[(index_x % BK)][(index_y % BM)] = regA[i];
            }
            __syncthreads();
        }
    }

    for (int iterWarpM = 0; iterWarpM < nbIterWarpM; iterWarpM++)
    {
        for (int iterWarpN = 0; iterWarpN < nbIterWarpN; iterWarpN++)
        {
            int xOut = blockIdx.x * BN + warpIdx * WN + iterWarpN * SUBWN + TN * idxInWarp;
            int yOut = blockIdx.y * BM + warpIdy * WM + iterWarpM * SUBWM + TM * idyInWarp;
            for (int yt = 0; yt < TM; yt++)
            {
                for (int xt = 0; xt < TN; xt++)
                {
                    int indexC = N * (yOut + yt) + xOut + xt;
                    c[indexC] = beta * c[indexC] + alpha * c_regs[TN * nbIterWarpN * (iterWarpM * TM + yt) + (iterWarpN * TN + xt)];
                }
            }
        }
    }
}
void Kernel5LdsOptim::init()
{
}
void Kernel5LdsOptim::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N)
{
    auto threadsPerBlock = dim3(BLOCK_SIZE);
    auto blocksPerGrid = dim3(N / 128, N / 128);
    hipLaunchKernelGGL(kernel5_lds_optim<1>, blocksPerGrid, threadsPerBlock, 0, 0, d_a, d_b, d_c, N, alpha, beta);
}

void Kernel5LdsOptim::finalize()
{
}
