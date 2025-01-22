#pragma once

class ISgemm
{
public:
    virtual ~ISgemm() = default;
    virtual std::string name() const = 0;
    virtual void init() = 0;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) = 0;
    virtual void finalize() = 0;
};