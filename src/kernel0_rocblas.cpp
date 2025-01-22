#include "kernel0_rocblas.h"
#include <iostream>
#include "sgemm.h"

#define CHECK_ROCBLAS_STATUS(status)                           \
    if (status != rocblas_status_success)                      \
    {                                                          \
        std::cerr << "rocBLAS error: " << status << std::endl; \
        exit(EXIT_FAILURE);                                    \
    }

void Kernel0ROCBLAS::init()
{
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));
}

void Kernel0ROCBLAS::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) 
{
    const int M = N;
    const int K = N;
    CHECK_ROCBLAS_STATUS(rocblas_sgemm(
        handle,
        rocblas_operation_none, // Transpose option for A
        rocblas_operation_none, // Transpose option for B
        M,                      // Number of rows in A and C
        N,                      // Number of columns in B and C
        K,                      // Number of columns in A and rows in B
        &alpha,                 // alpha
        d_a,                    // Matrix A on the device
        M,                      // Leading dimension of A
        d_b,                    // Matrix B on the device
        K,                      // Leading dimension of B
        &beta,                  // beta
        d_c,                    // Matrix C on the device
        M                       // Leading dimension of C
        ));
}
void Kernel0ROCBLAS::finalize() 
{
    rocblas_destroy_handle(handle);
}