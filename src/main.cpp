#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel0_rocblas.h"
#include "kernel1_naive.h"
#include "kernel2_lds.h"
#include "kernel3_registers.h"
#include "kernel4_gmem_db.h"
#include "kernel5_lds_optim.h"
#include "kernel6_valu_optim.h"
#include "kernel7_unroll.h"
#include "kernel8_batched_gmem.h"
#include <iostream>
#include <vector>

#define CHECK_HIP_STATUS(status)                           \
    if (status != hipSuccess)                              \
    {                                                      \
        std::cerr << "hip error: " << status << std::endl; \
        exit(EXIT_FAILURE);                                \
    }

namespace
{

    float getBestTiming(hipEvent_t *start, hipEvent_t *stop, int N)
    {
        float minTime = std::numeric_limits<float>::max();
        float sum = 0;
        for (int i = 0; i < N; i++)
        {
            float elapsedTime = 0.0;
            CHECK_HIP_STATUS(hipEventElapsedTime(&elapsedTime, start[i], stop[i]));
            minTime = std::min(minTime, elapsedTime);
        }
        return minTime;
    }

    float timeToGFlops(float time_ms, float N)
    {
        return 1e-6 * (N * N * N * 2 + 3 * N * N) / time_ms;
    }

    void test_sgemm(ISgemm &sgemm)
    {
        sgemm.init();
        const int N = 4096;
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        float *h_A, *h_B, *h_C; // Host vectors
        float *d_A, *d_B, *d_C; // Device vectors

        // Allocate memory on the host
        h_A = (float *)malloc(N * N * sizeof(float));
        h_B = (float *)malloc(N * N * sizeof(float));
        h_C = (float *)malloc(N * N * sizeof(float));

        for (int i = 0; i < N * N; i++)
        {
            h_A[i] = static_cast<float>(i);
        }

        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                h_B[N * y + x] = x == y ? 1.0f : 0.0f;
            }
        }

        constexpr int nbRun = 16;

        hipEvent_t start[nbRun], stop[nbRun];
        for (int i = 0; i < nbRun; i++)
        {
            CHECK_HIP_STATUS(hipEventCreate(start + i));
            CHECK_HIP_STATUS(hipEventCreate(stop + i));
        }
        // Allocate memory on the device
        CHECK_HIP_STATUS(hipMalloc((void **)&d_A, N * N * sizeof(float)));
        CHECK_HIP_STATUS(hipMalloc((void **)&d_B, N * N * sizeof(float)));
        CHECK_HIP_STATUS(hipMalloc((void **)&d_C, N * N * sizeof(float)));

        // Copy vectors from host to device
        CHECK_HIP_STATUS(hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice));

        for (int i = 0; i < nbRun; i++)
        {
            CHECK_HIP_STATUS(hipEventRecord(start[i], 0));
            sgemm.run(d_A, d_B, d_C, alpha, beta, N);
            CHECK_HIP_STATUS(hipEventRecord(stop[i], 0));
        }

        CHECK_HIP_STATUS(hipDeviceSynchronize());

        auto bestTiming = getBestTiming(start, stop, nbRun);
        std::cout << bestTiming << " ms -> " << timeToGFlops(bestTiming, N) << " GFLOPS" << std::endl;

        // Copy result vector from device to host
        CHECK_HIP_STATUS(hipMemcpy(h_C, d_C, N * N * sizeof(float), hipMemcpyDeviceToHost));

        // Validate the result
        for (int i = 0; i < N * N; i++)
        {
            if (h_C[i] != h_A[i])
            {
                std::cout << "Error at index " << i << ": " << h_C[i] << " vs " << h_A[i] << std::endl;
                break;
            }
        }

        // Free device memory
        CHECK_HIP_STATUS(hipFree(d_A));
        CHECK_HIP_STATUS(hipFree(d_B));
        CHECK_HIP_STATUS(hipFree(d_C));

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);
        sgemm.finalize();
    }

}
int main()
{

    std::vector<std::unique_ptr<ISgemm>> kernels;
    kernels.push_back(std::make_unique<Kernel0ROCBLAS>());
    kernels.push_back(std::make_unique<Kernel1Naive>());
    kernels.push_back(std::make_unique<Kernel2Lds>());
    kernels.push_back(std::make_unique<Kernel3Registers>());
    kernels.push_back(std::make_unique<Kernel4GmemDB>());
    kernels.push_back(std::make_unique<Kernel5LdsOptim>());
    kernels.push_back(std::make_unique<Kernel6VALUOptim>());
    kernels.push_back(std::make_unique<Kernel7Unroll>());
    kernels.push_back(std::make_unique<Kernel8BatchedGMem>());
    for (auto &k : kernels)
    {
        std::cout << k->name() << std::endl;
        test_sgemm(*k);
        std::cout << "--------------------" << std::endl;
    }
    return 0;
}