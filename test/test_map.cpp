#include<iostream>
#include<vector>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include"arcticfox/map.h"

static const int len = 1024;

int main(){
    std::vector<float> A(len);
    std::vector<float> B(len);
    std::vector<float> C(len);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 1.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    float *Ad, *Bd, *Cd;
    hipMalloc(&Ad, len*sizeof(float));
    hipMalloc(&Bd, len*sizeof(float));
    hipMalloc(&Cd, len*sizeof(float));

    hipMemcpy(Ad, A.data(), len*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Bd, B.data(), len*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Cd, C.data(), len*sizeof(float), hipMemcpyHostToDevice);

    arcticfox::plus<float> op;
    arcticfox::map(Cd, Bd, Ad, op, 512);
}

