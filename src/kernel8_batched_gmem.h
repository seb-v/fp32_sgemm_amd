#pragma once
class ISgemm;
#include <hip/hip_runtime.h>

class Kernel8BatchedGMem : public ISgemm
{
    virtual std::string name() const override
    {
        return "Kernel 8 : Batched GMem loads";
    }
    virtual void init() override;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    virtual void finalize() override;

private:
    hipModule_t _module;
    hipFunction_t _kernelFunc;
};

