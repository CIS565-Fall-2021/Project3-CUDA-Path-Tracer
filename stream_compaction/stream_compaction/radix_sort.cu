#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cVec.h"

#define block_size 128

namespace Radix {

	__global__ void kern_bitcmp(const int *x, int n, int b, int *bools) {
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;
		bools[idx] = 1 - ((x[idx] >> b) & 1);
	}

	/* assumes lenght of bscan, N, is longer than n */
	__global__ void kern_scatter(int *x, int *bools, int* bscan, int n, int *out)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n)
			return;

		int out_idx = bools[idx] ? bscan[idx] : (bools[n-1]+bscan[n-1]) + idx - bscan[idx];
		out[out_idx] = x[idx];
	}
	
	/* n length of x, bools, and out, N length of scanned bools */
	void split(cu::cVec<int> *x, cu::cVec<int> *out, cu::cVec<int> *bools, int n,
		cu::cVec<int> *bscan, int N, int b)
	{
		int blocks_per_grid = ((n + block_size - 1) / block_size);

		kern_bitcmp<<<blocks_per_grid, block_size>>>(x->raw_ptr(), n, b, bools->raw_ptr());
	
		cu::copy(bscan->ptr(), bools->ptr(), n);

		StreamCompaction::Efficient::scan_dev(N, bscan);
		
		kern_scatter<<<blocks_per_grid, block_size>>>(x->raw_ptr(), bools->raw_ptr(), bscan->raw_ptr(), n, out->raw_ptr());
	}

	void sort(int *x, int n)
	{
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;
		
		cu::cVec<int> dev_x(n, x);
		cu::cVec<int> dev_bools(n);
		cu::cVec<int> dev_bscan(N); // scanned bools
		cu::cVec<int> dev_out(n);

		StreamCompaction::Efficient::timer().startGpuTimer();
		for (int b = 0; b < 32; b++) {
			split(&dev_x, &dev_out, &dev_bools, n, &dev_bscan, N, b);
			std::swap(dev_x, dev_out);
		}
		StreamCompaction::Efficient::timer().endGpuTimer();

		cu::copy(x, dev_x.ptr(), n);
	}
}
