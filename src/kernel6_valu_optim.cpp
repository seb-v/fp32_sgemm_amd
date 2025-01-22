#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel6_valu_optim.h"

void Kernel6VALUOptim::init()
{
    // Load the .hsaco file
    if (hipSuccess != hipModuleLoad(&_module, "kernel6.hsaco"))
    {
        throw std::runtime_error("Error loading kernel6.hscao");
    }

    // Get the function handle
    if (hipSuccess != hipModuleGetFunction(&_kernelFunc, _module, "kernel"))
    {

        throw std::runtime_error("Error getting function from hscao file");
    }
}

void Kernel6VALUOptim::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N)
{
    void *args[] = {&d_a, &d_b, &d_c, &N, &alpha, &beta};

    if (hipSuccess != hipModuleLaunchKernel(
                          _kernelFunc,         // Kernel function
                          N / 128, N / 128, 1, // Grid dimensions
                          128, 1, 1,           // Block dimensions
                          0,                   // Shared memory size
                          nullptr,             // Stream
                          (void **)&args,      // kernelParams
                          nullptr))
    {

        throw std::runtime_error("Failed to launch kernel");
    }
}

void Kernel6VALUOptim::finalize()
{
    if (hipSuccess !=  hipModuleUnload(_module))
    {
        throw std::runtime_error("Failed to unload hip module");
    }
}
