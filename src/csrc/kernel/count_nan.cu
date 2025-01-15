#include "count_nan.h"

#include <cassert>
#include <cstdint>

#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T>
__device__ bool is_inf(T val) {
	if constexpr (std::is_same<T, float>::value) {
		return ((uint32_t*)(&val))[0] == 0x7f800000;
	} else if constexpr (std::is_same<T, half>::value) {
		return ((uint16_t*)(&val))[0] == 0x7c00;
	} else {
		assert(false);
	}
}

template<typename T>
__global__ void countNanKernel(
	int* count,
	const T* arr,
	int n
) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		if (arr[i] != arr[i] || is_inf<T>(arr[i]) || is_inf<T>(-arr[i])) {
			atomicAdd(count, 1);
		}
	}
}

template<typename T>
int countNan(
	const T* arr,
	int n
) {
	int* count;
	cudaMalloc(&count, sizeof(int));
	cudaMemset(count, 0, sizeof(int));

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	countNanKernel<T><<<numBlocks, blockSize>>> (count, arr, n);

	int res;
	cudaMemcpy(&res, count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(count);
	return res;
}

#define INSTANTIATE(T) \
	template int countNan<T>(const T* arr, int n);

INSTANTIATE(float)
INSTANTIATE(half)

}
