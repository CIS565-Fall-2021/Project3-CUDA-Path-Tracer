#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
namespace CPU {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	/**
	 * CPU scan (prefix sum).
	 * For performance analysis, this is supposed to be a simple for loop.
	 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
	 */
	void scan(int n, int *odata, const int *idata)
	{
		timer().startCpuTimer();

		odata[0] = 0;
		for (int i = 1; i < n; i++)
			odata[i] = odata[i-1] + idata[i-1];

		timer().endCpuTimer();
	}

	/**
	 * CPU stream compaction without using the scan function.
	 *
	 * @returns the number of elements remaining after compaction.
	 */
	int compactWithoutScan(int n, int *odata, const int *idata)
	{
		timer().startCpuTimer();
		
		const int *in = idata;
		int *out = odata;
		const int *in_end = idata + n;

		for (; in < in_end; in++) {
			if (*in)
				*out++ = *in;
		}

		timer().endCpuTimer();
		return out - odata;
	}

	/**
	 * CPU stream compaction using scan and scatter, like the parallel version.
	 *
	 * @returns the number of elements remaining after compaction.
	 */
	int compactWithScan(int n, int *odata, const int *idata)
	{
		int* bdata = new int[n];
		int* sdata = new int[n];
		int count = 0;

		timer().startCpuTimer();

		for (int i = 0; i < n; i++)
			bdata[i] = idata[i] ? 1 : 0;

		/* scan */
		sdata[0] = 0;
		for (int i = 1; i < n; i++)
			sdata[i] = sdata[i - 1] + bdata[i - 1];

		/* scatter */
		for (int i = 0; i < n; i++) {
			if (bdata[i]) {
				odata[sdata[i]] = idata[i];
				count++;
			}
		}

		timer().endCpuTimer();

		delete[] bdata;
		delete[] sdata;
		return count;
	}
}
}
