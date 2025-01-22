#pragma once
class ISgemm;

class Kernel1Naive : public ISgemm
{
    virtual std::string name() const override {
        return "Kernel 1 : Naive";
    }
    virtual void init() override;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    virtual void finalize() override;
};
