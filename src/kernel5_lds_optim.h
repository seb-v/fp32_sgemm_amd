#pragma once
class ISgemm;

class Kernel5LdsOptim : public ISgemm
{
    virtual std::string name() const override
    {
        return "Kernel 5 :  LDS Utilization Optimization";
    }
    virtual void init() override;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    virtual void finalize() override;
};
