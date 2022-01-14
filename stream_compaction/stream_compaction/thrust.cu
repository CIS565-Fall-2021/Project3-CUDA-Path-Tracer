#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}
	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void scan(int n, int *odata, const int *idata)
	{
		thrust::host_vector<int> host_idata(idata, idata + n);
		thrust::device_vector<int> dv_in = host_idata;
		thrust::device_vector<int> dv_out(n);

		timer().startGpuTimer();
		printf("here\n");
		thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
		printf("here\n");
		timer().endGpuTimer();

		thrust::copy(dv_out.begin(), dv_out.end(), odata);
	}


	struct is_zero {
		__host__ __device__ bool operator()(int n) {
			return n == 0;
		}
	};

	/**
	 * Performs stream compaction on idata, removing 0 values and storing the result into odata.
	 */
	int compact(int n, int* odata, const int* idata)
	{
		thrust::host_vector<int> host_idata(idata, idata + n);
		thrust::device_vector<int> dev_data = host_idata;

		timer().startGpuTimer();
		auto new_end = thrust::remove_if(dev_data.begin(), dev_data.end(), is_zero());
		timer().endGpuTimer();

		thrust::copy(dev_data.begin(), new_end, odata);

		return new_end - dev_data.begin();
	}
}
}
