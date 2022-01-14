#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "cVec.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 256

namespace StreamCompaction {
namespace Naive {


	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	template <typename T>
	__global__ void kern_scan(int d, int n, const T *__restrict__ in, T *__restrict__ out)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;

		if (idx >= (1 << d))
			out[idx] = in[idx] + in[idx - (1 << d)];
		else
			out[idx] = in[idx];
	}



	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int* odata, const int* idata)
	{
		/* the implementation does inclusive scan, then the first (n-1) vals are copied
		 * to odata[1:] and odata[0] is set to 0
		*/

		cu::cVec<int> data_in(n, idata), data_out(n);
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		timer().startGpuTimer();

		int log2n = ilog2ceil(n);
		for (int d = 0; d < log2n; d++) {
			kern_scan<<<fullBlocksPerGrid, blockSize>>>(d, n, data_in.raw_ptr(), data_out.raw_ptr());
			checkCUDAError("kern_scan failed!");
			std::swap(data_in, data_out);
		}

		timer().endGpuTimer();


		cu::copy(odata+1, data_in.ptr(), n-1);
		odata[0] = 0;
	}

	
	template <typename T>
	__global__ void kern_shared_scan(int n, const T* __restrict__ in, T* __restrict__ out)
	{
		extern __shared__ T tmp[];
		T *b1 = tmp, *b2 = tmp + blockSize, *b3; /* double buffers for swapping */
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;

		b1[idx] = idx > 0 ? in[idx-1] : 0;
		__syncthreads();
		
		#pragma unroll
		for (int d = 1; d < blockSize; d *= 2) { /* p is for swapping between shared mem */
			b3 = b1, b2 = b1, b2 = b3; /* swap b1, b2*/
			if (idx >= d)
				b1[idx] += b2[idx - d];
			else
				b1[idx] = b2[idx];
			__syncthreads();
		}
		out[idx] = b1[idx];
	}

	template <typename T>
	__global__ void kern_add(int n, T* x, const T* val)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;
		
		x[idx] += *val;
	}


	void shared_scan(int n, int* odata, const int* idata)
	{
		int block_count = (n+blockSize - 1) / blockSize;
		int N = block_count * blockSize;
	
		cu::cVec<int> in(n, idata, N), out(N);

		timer().startGpuTimer();
		
		kern_shared_scan<<<block_count, blockSize, 2 * blockSize * sizeof(int)>>>(n, in.raw_ptr(), out.raw_ptr());
		
		for (int i = 1; i < block_count; i++) {
			kern_shared_scan<<<block_count - i, blockSize>>>(n - i * blockSize, out.raw_ptr() + blockSize * i, out.raw_ptr() + blockSize * i - 1);
		}

		timer().endGpuTimer();

		cu::copy(odata, out.ptr(), n);
	}
}
}

