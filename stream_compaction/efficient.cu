#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)  // We can use defines provided in this project

		int* dev_buf;
		int* dev_bufloader;
		int* dev_bufB;
		int* dev_bufS;
		int* dev_bufAnswers;

		PathSegment* dev_PTbuf;
		PathSegment* dev_PTbufAnswers;

#define blockSize 512

		__global__ void performUpSweep(int d, int* buf, int N)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pow2_d = 1 << d;
			int pow2_dplus1 = 1 << (d + 1);
			if (index + pow2_dplus1 - 1 > N - 1)
			{
				return;
			}

			if ((index + pow2_dplus1 - 1) % pow2_dplus1 == (pow2_dplus1 - 1))
			{
				buf[index + pow2_dplus1 - 1] += buf[index + pow2_d - 1];
			}

			if (index + pow2_dplus1 - 1 == N - 1)
			{
				buf[index + pow2_dplus1 - 1] = 0;
				return;
			}

		}

		__global__ void performDownSweep(int d, int* buf, int N)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pow2_d = 1<<d;
			int pow2_dplus1 = 1<<(d+1);
			if (index + pow2_dplus1 - 1 > N - 1)
			{
				return;
			}

			if ((index + pow2_dplus1 - 1) % pow2_dplus1 == (pow2_dplus1 - 1))
			{

				int t = buf[index + pow2_d - 1];
				buf[index + pow2_d - 1] = buf[index + pow2_dplus1 - 1];
				buf[index + pow2_dplus1 - 1] += t;
			}
		}

		void AllocateMemory(int n)
		{
			cudaMalloc((void**)&dev_buf, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
			cudaMalloc((void**)&dev_bufloader, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufloader failed!");
			cudaMalloc((void**)&dev_bufB, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufB failed!");
			cudaMalloc((void**)&dev_bufS, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufS failed!");
			cudaMalloc((void**)&dev_bufAnswers, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufAnswers failed!");	
			cudaDeviceSynchronize();
		}

		void AllocateMemoryPathTracer(int n)
		{
			cudaMalloc((void**)&dev_PTbuf, n * sizeof(PathSegment));
			checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
			cudaMalloc((void**)&dev_bufloader, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufloader failed!");
			cudaMalloc((void**)&dev_bufB, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufB failed!");
			cudaMalloc((void**)&dev_bufS, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufS failed!");
			cudaMalloc((void**)&dev_PTbufAnswers, n * sizeof(PathSegment));
			checkCUDAErrorWithLine("cudaMalloc dev_bufAnswers failed!");
			cudaDeviceSynchronize();
		}

		void FreeMemory() {
			cudaFree(dev_buf);
			cudaFree(dev_bufloader);
			cudaFree(dev_bufB);
			cudaFree(dev_bufS);
			cudaFree(dev_bufAnswers);
			cudaFree(dev_PTbuf);
			cudaFree(dev_PTbufAnswers);
			
		}

		__global__ void RightShiftAddZeros(int* buf,int *buf_loader, int N, int difference)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			if (index < difference)
			{
				buf[index] = 0;
				return;
			}
			buf[index] = buf_loader[index-difference];
		}

		__global__ void RightShiftDeleteZeroes(int* buf, int* buf_loader, int N, int difference)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index > N - 1)
			{
				return;
			}
			buf[index] = buf_loader[index + difference];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			// TODO

			int nearesttwo = 1<<ilog2ceil(n);

			int difference = nearesttwo - n;

			int finalMemSize = nearesttwo;

			AllocateMemory(finalMemSize);


			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);
			cudaMemcpy(dev_bufloader, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			RightShiftAddZeros << < fullBlocksPerGrid, blockSize >> > (dev_buf, dev_bufloader, finalMemSize, difference);
		
			int d = ilog2ceil(finalMemSize);

			for (int i = 0; i <= d - 1; i++)
			{
				performUpSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}

		
			for (int i = d - 1; i >= 0; i--)
			{
				performDownSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf, finalMemSize);
			}

			RightShiftDeleteZeroes << < fullBlocksPerGrid, blockSize >> > (dev_bufloader, dev_buf, n, difference);
			cudaDeviceSynchronize();
			cudaMemcpy(odata, dev_bufloader, sizeof(int) * n, cudaMemcpyDeviceToHost);
	
			timer().endGpuTimer();

			FreeMemory();
			cudaDeviceSynchronize();
		}

		int* dev_buf2;
		int* dev_bufloader2;

		void AllocateMemory2(int n)
		{
			cudaMalloc((void**)&dev_buf2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_buf failed!");
			cudaMalloc((void**)&dev_bufloader2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bufloader failed!");
			
		}

		void FreeMemory2() {
			cudaFree(dev_buf2);
			cudaFree(dev_bufloader2);
		}


		void scanWithoutTimer(int n, int* odata, const int* idata) {
			// TODO
			int nearesttwo = 1 << ilog2ceil(n);

			int difference = nearesttwo - n;

			int finalMemSize = nearesttwo;

			AllocateMemory2(finalMemSize);
			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);
			cudaMemcpy(dev_bufloader2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			RightShiftAddZeros << < fullBlocksPerGrid, blockSize >> > (dev_buf2, dev_bufloader2, finalMemSize, difference);

			int d = ilog2ceil(finalMemSize);

			for (int i = 0; i <= d - 1; i++)
			{
				performUpSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf2, finalMemSize);
			}

			for (int i = d - 1; i >= 0; i--)
			{
				performDownSweep << < fullBlocksPerGrid, blockSize >> > (i, dev_buf2, finalMemSize);
			}

			RightShiftDeleteZeroes << < fullBlocksPerGrid, blockSize >> > (dev_bufloader2, dev_buf2, n, difference);
			cudaMemcpy(odata, dev_bufloader2, sizeof(int) * n, cudaMemcpyDeviceToHost);
			FreeMemory2();
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int* odata, const int* idata) {
			int finalMemSize = n;
			AllocateMemory(finalMemSize);


			timer().startGpuTimer();
			// TODO
			cudaMemcpy(dev_buf, idata, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);


			Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_bufB, dev_buf);

			//Copy bool value to new array to perform scan
			int* arr_boolean = new int[finalMemSize];
			cudaMemcpy(arr_boolean, dev_bufB, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

				   //Create new array to store answers from scan
			int* arr_scanResult = new int[finalMemSize];
			scanWithoutTimer(finalMemSize, arr_scanResult, arr_boolean);

				//Copy the scan answers to Dev_BufS to further process
			cudaMemcpy(dev_bufS, arr_scanResult, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			int numElements = arr_scanResult[finalMemSize - 1];

			Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_bufAnswers, dev_buf,
				dev_bufB, dev_bufS);

			cudaMemcpy(odata, dev_bufAnswers, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

			timer().endGpuTimer();
			FreeMemory();

			if (arr_boolean[finalMemSize - 1] == 1)
			{
				return numElements + 1; // Since indexing start from 0
			}
			return numElements; //if last element boolean is 0 its scan result include 1 extra sum counting for 0 index
		}


		int compact(int n, PathSegment* dev_iPathSegment) {
			int finalMemSize = n;
			AllocateMemoryPathTracer(finalMemSize);


			//timer().startGpuTimer();
			// TODO
			cudaMemcpy(dev_PTbuf, dev_iPathSegment, sizeof(PathSegment) * finalMemSize, cudaMemcpyDeviceToDevice);

			dim3 fullBlocksPerGrid((finalMemSize + blockSize - 1) / blockSize);


			Common::kernMapToBooleanPathTracer << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_bufB, dev_PTbuf);

			//Copy bool value to new array to perform scan
			int* arr_boolean = new int[finalMemSize];
			cudaMemcpy(arr_boolean, dev_bufB, sizeof(int) * finalMemSize, cudaMemcpyDeviceToHost);

			//Create new array to store answers from scan
			int* arr_scanResult = new int[finalMemSize];
			scanWithoutTimer(finalMemSize, arr_scanResult, arr_boolean);

			//Copy the scan answers to Dev_BufS to further process
			cudaMemcpy(dev_bufS, arr_scanResult, sizeof(int) * finalMemSize, cudaMemcpyHostToDevice);

			int numElements = arr_scanResult[finalMemSize - 1];

			Common::kernScatterPathTracer << < fullBlocksPerGrid, blockSize >> > (finalMemSize, dev_PTbufAnswers, dev_PTbuf,
				dev_bufB, dev_bufS);

			cudaMemcpy(dev_iPathSegment, dev_PTbufAnswers, sizeof(PathSegment) * finalMemSize, cudaMemcpyDeviceToDevice);

			//timer().endGpuTimer();
			FreeMemory();

			if (arr_boolean[finalMemSize - 1] == 1)
			{
				return numElements + 1; // Since indexing start from 0
			}
			return numElements; //if last element boolean is 0 its scan result include 1 extra sum counting for 0 index
		}
	}
}
