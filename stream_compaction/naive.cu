#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
namespace StreamCompaction {
	namespace Naive {

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)  // We can use defines provided in this project


		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		int* dev_buf1;
		int* dev_buf2;
		int* dev_bufLoader;
#define blockSize 512

		__global__ void performScan(int d, int* buf1, int* buf2, int N)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			
			//int pow2_dminus1 = std::round(pow(2, d - 1));
			int pow2_dminus1 = 1 <<(d - 1);
			if (index >= pow2_dminus1)
			{
				buf2[index] = buf1[index - pow2_dminus1] + buf1[index];
			}
			else
			{
				buf2[index] = buf1[index];
			}

		}

		__global__ void ShiftRight(int* buf1, int* buf2, int N, int difference)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			if (index == 0)
			{
				buf2[index] = 0;
				return;
			}
			buf2[index] = buf1[index - 1];

		}

		void FreeMemory() {
			cudaFree(dev_buf1);
			cudaFree(dev_buf2);
			cudaFree(dev_bufLoader);
		}

		void AllocateMemory(int n)
		{
			cudaMalloc((void**)&dev_buf1, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf1 failed!");
			cudaMalloc((void**)&dev_buf2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf2 failed!");
			cudaMalloc((void**)&dev_bufLoader, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufLoader failed!");
			cudaDeviceSynchronize();
		}

		__global__ void RightShiftAddZeros(int* buf, int* buf_loader, int N, int difference)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			if (index > (N-1) - difference)
			{
				buf[index] = 0;
				return;
			}
			buf[index] = buf_loader[index];
		}


		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			// TODO

			int power2 = 1;
			int nearesttwo = 1 << ilog2ceil(n);

			int difference = nearesttwo - n;

			int finalMemSize = nearesttwo;


			AllocateMemory(finalMemSize);


			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);
			dim3 threadsperblockSize(blockSize);


			cudaMemcpy(dev_bufLoader, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			RightShiftAddZeros << < fullBlocksPerGrid, threadsperblockSize >> > (dev_buf1, dev_bufLoader, finalMemSize, difference);


			int d = ilog2(finalMemSize);
			for (int i = 1; i <= d; i++)
			{
				performScan << < fullBlocksPerGrid, threadsperblockSize >> > (i, dev_buf1, dev_buf2, finalMemSize);
				//cudaDeviceSynchronize();
				std::swap(dev_buf1, dev_buf2);
			}
			
			ShiftRight << < fullBlocksPerGrid, blockSize >> > (dev_buf1, dev_buf2, finalMemSize, difference);
			cudaMemcpy(odata, dev_buf2, sizeof(int) * n, cudaMemcpyDeviceToHost);

		
			/*printf(" \n Array After:");*/
			/*for (int i = 0; i < finalMemSize; i++)
			{
				printf("%3d ", arr_z[i]);
			}*/

			/*	 printf("]\n");
				 for (int i = 0; i < n; i++)
				 {
					 printf("%3d ", odata[i]);
				 }*/


			timer().endGpuTimer();
			cudaDeviceSynchronize();
			FreeMemory();
		}
	}
}
