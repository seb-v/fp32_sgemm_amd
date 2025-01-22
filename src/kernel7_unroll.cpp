#include <hip/hip_runtime.h>
#include "sgemm.h"
#include "kernel7_unroll.h"

void Kernel7Unroll::init()
{
    // Load the .hsaco file
    if (hipSuccess != hipModuleLoad(&_module, "kernel7.hsaco"))
    {
        throw std::runtime_error("Error loading kernel7.hscao");
    }

    // Get the function handle
    if (hipSuccess != hipModuleGetFunction(&_kernelFunc, _module, "kernel"))
    {

        throw std::runtime_error("Error getting function from hscao file");
    }
}

void Kernel7Unroll::run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N)
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

void Kernel7Unroll::finalize()
{
    if (hipSuccess !=  hipModuleUnload(_module))
    {
        throw std::runtime_error("Failed to unload hip module");
    }
}

