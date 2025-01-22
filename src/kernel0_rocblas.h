#pragma once
#include <rocblas/rocblas.h>
#include "sgemm.h"

class Kernel0ROCBLAS : public ISgemm
{
    rocblas_handle handle;
    virtual std::string name() const override {
        return "Kernel 0 : ROCBLAS";
    }
    virtual void init() override;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    virtual void finalize() override;
};