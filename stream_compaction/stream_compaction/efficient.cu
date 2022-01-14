#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cVec.h"

#define blockSize 256

namespace StreamCompaction {
namespace Efficient {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	template <typename T>
	__global__ void kern_up_sweep(int n, T* x, int stride)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n || idx > n/stride/2)
			return;

//		for (int stride = 1; stride < n; stride *= 2) {

			int src = idx * stride * 2 + stride - 1;
			int dst = src + stride;

			if (dst >= n)
				return;

			x[dst] += x[src];
//			__syncthreads();
//		}
	}


	template <typename T>
	__global__ void kern_down_sweep(int n, T* x, int stride)
	{
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= n || idx > n/stride/2)
			return;

//		for (int stride = n/2; stride >= 1; stride /= 2) {

			int src = idx * stride * 2 + stride - 1;
			int dst = src + stride;
	
			if (dst < n) {
				int src_val = x[src];
				x[src] = x[dst];
				x[dst] += src_val;
			}
//			__syncthreads();
//		}
	}

	/* in-place scan over device array, doesn't start GPU Timer and assumes input is power of 2
	 * the zero value of T is assumed to be (T) 0
	 */
	template <typename T>
	void scan_dev(int N, cu::cVec<T>* dev_data) {
		int blocks_per_grid;

		for (int stride = 1; stride < N; stride *= 2) {
			blocks_per_grid = (N/stride/2 + blockSize - 1) / blockSize;
			kern_up_sweep<<<blocks_per_grid, blockSize>>>(N, dev_data->raw_ptr(), stride);
		}
		cu::set(dev_data->ptr() + N-1, (T) 0, 1);
		for (int stride = N/2; stride >= 1; stride /= 2) {
			blocks_per_grid = (N/stride/2 + blockSize - 1) / blockSize;
			kern_down_sweep<<<blocks_per_grid, blockSize>>>(N, dev_data->raw_ptr(), stride);
		}
	}

	/* the scan_dev template is not exported in the header and must explicitly be instantiated */
	template void scan_dev<int>(int, cu::cVec<int>*);
	template void scan_dev<size_t>(int, cu::cVec<size_t>*);


	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cu::cVec<int> dev_data(n, idata, N);

		timer().startGpuTimer();

		scan_dev(N, &dev_data);

		timer().endGpuTimer();

		cu::copy(odata, dev_data.ptr(), n);
	}

	/**
	 * Performs stream compaction on idata, storing the result into odata.
	 * All zeroes are discarded.
	 *
	 * @param n      The number of elements in idata.
	 * @param odata  The array into which to store elements.
	 * @param idata  The array of elements to compact.
	 * @returns	The number of elements remaining after compaction.
	 */
	int compact(int n, int *odata, const int *idata) {
		int log2n = ilog2ceil(n);
		int N = 1 << log2n;

		cu::cVec<int> dev_idata(n, idata), dev_bdata(n), dev_sdata(N), dev_odata(n);

		timer().startGpuTimer();
		
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bdata.raw_ptr(), dev_idata.raw_ptr());

		cu::copy(dev_sdata.ptr(), dev_bdata.ptr(), n);
		scan_dev(N, &dev_sdata);

		Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata.raw_ptr(), dev_idata.raw_ptr(), dev_bdata.raw_ptr(), dev_sdata.raw_ptr());

		timer().endGpuTimer();

		cu::copy(odata, dev_odata.ptr(), n);

		for (int i = 0; i < n; i++)
			if (!odata[i])
				return i;
		
		return n;
	}
}
}
