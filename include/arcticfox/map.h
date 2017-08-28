#pragma once

namespace arcticfox {
template<typename T>
struct plus {
//    __host__ __device__ plus() {}
//    __host__ __device__ ~plus() {}
    __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
        return lhs + rhs;
    }
};
}


namespace arcticfox {
namespace kernel {
template<typename T, typename Func>
__global__ void Map(T* Out, T* In1, T* In2, Func func) {
    int tid = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    Out[tid] = func(In1[tid], In2[tid]);
}
}
}

namespace arcticfox {
template<typename T, typename Func>
void map(T *Out, T *In1, T *In2, Func func, size_t len) {
    hipLaunchKernelGGL((kernel::Map<T, Func>), dim3(1,1,1), dim3(len,1,1), 0, 0, Out, In1, In2, func);
}
}
