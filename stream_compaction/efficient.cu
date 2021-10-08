#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define USING_SHARED_MEMORY 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#if USING_SHARED_MEMORY

        const int blockSize_sharedMemory = 128;

        const int logNumBanks = 5;

        #define CONFLICT_FREE_OFFSET(n) ((n) >> logNumBanks)

        __global__ void kernScanPerBlock(int n, int *dev_data, int *dev_blockSum) {

            if (n == 1) {
                dev_data[0] = 0;
                return;
            }
            if (threadIdx.x >= n / 2) {
                return;
            }
            
            __shared__ int temp[2 * blockSize_sharedMemory + CONFLICT_FREE_OFFSET(2 * blockSize_sharedMemory)];

            dev_data += blockDim.x * blockIdx.x * 2;
            if (n > blockDim.x * 2) {
                n = blockDim.x * 2;
            }

            int i = threadIdx.x;
            int j = threadIdx.x + n / 2;
            int ti = i + CONFLICT_FREE_OFFSET(i);
            int tj = j + CONFLICT_FREE_OFFSET(j);
            temp[ti] = dev_data[i];
            temp[tj] = dev_data[j];
            
            int lastElement = 0;
            if (dev_blockSum && threadIdx.x == blockDim.x - 1) {
                lastElement = temp[tj];
            }

            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (threadIdx.x < d) {
                    int i = offset * (2 * threadIdx.x + 1) - 1;
                    int j = offset * (2 * threadIdx.x + 2) - 1;
                    i += CONFLICT_FREE_OFFSET(i);
                    j += CONFLICT_FREE_OFFSET(j);
                    temp[j] += temp[i];
                }
                offset *= 2;
            }

            if (threadIdx.x == 0) {
                temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
            }

            for (int d = 1; d < n; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (threadIdx.x < d) {
                    int i = offset * (2 * threadIdx.x + 1) - 1;
                    int j = offset * (2 * threadIdx.x + 2) - 1;
                    i += CONFLICT_FREE_OFFSET(i);
                    j += CONFLICT_FREE_OFFSET(j);
                    int t = temp[i];
                    temp[i] = temp[j];
                    temp[j] += t;
                }
            }

            __syncthreads();
            dev_data[i] = temp[ti];
            dev_data[j] = temp[tj];

            if (dev_blockSum && threadIdx.x == blockDim.x - 1) {
                dev_blockSum[blockIdx.x] = lastElement + temp[tj];
            }
        }

        __global__ void kernAddPerBlock(int *dev_data, int *dev_add) {
            int blockSum = dev_add[blockIdx.x];
            dev_data += blockIdx.x * blockDim.x * 2;
            dev_data[threadIdx.x] += blockSum;
            dev_data[threadIdx.x + blockDim.x] += blockSum;
        }

        void scanHelper(int size, int *dev_data) {

            if (size > 2 * blockSize_sharedMemory) {
                
                int blocks = size / (2 * blockSize_sharedMemory);
                int *dev_blockSum;
                cudaMalloc((void**) &dev_blockSum, blocks * sizeof(int));

                kernScanPerBlock<<<blocks, blockSize_sharedMemory>>>(size, dev_data, dev_blockSum);
                scanHelper(blocks, dev_blockSum);
                kernAddPerBlock<<<blocks, blockSize_sharedMemory>>>(dev_data, dev_blockSum);

                cudaFree(dev_blockSum);
                
            } else {
                kernScanPerBlock<<<1, blockSize_sharedMemory>>>(size, dev_data, nullptr);
            }
        }

#else

        __global__ void kernScanUpSweepPhase(int threads, int *dev_temp, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= threads) {
                return;
            }
            index = (index + 1) * offset * 2 - 1;
            dev_temp[index] += dev_temp[index - offset];
        }

        __global__ void kernScanDownSweepPhase(int threads, int *dev_temp, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= threads) {
                return;
            }
            index = (index + 1) * offset * 2 - 1;
            int t = dev_temp[index - offset];
            dev_temp[index - offset] = dev_temp[index];
            dev_temp[index] += t;
        }

        void scanHelper(int size, int *dev_temp) {

            int threads = size / 2;
            int offset = 1;
            for (; threads > 0; threads /= 2, offset *= 2) {
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernScanUpSweepPhase<<<blocks, blockSize>>>(threads, dev_temp, offset);
            }

            cudaMemset(dev_temp + size - 1, 0, sizeof(int));
            threads = 1;
            offset = size / 2;
            for (; offset > 0; offset /= 2, threads *= 2) {
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernScanDownSweepPhase<<<blocks, blockSize>>>(threads, dev_temp, offset);
            }
        }

#endif

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int size = 1 << ilog2ceil(n);

            int *dev_temp;
            cudaMalloc((void**) &dev_temp, size * sizeof(int));
            cudaMemcpy(dev_temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_temp + n, 0, (size - n) * sizeof(int));

            timer().startGpuTimer();
            scanHelper(size, dev_temp);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_temp, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_temp);
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
        int compact(int n, int *odata, const int *idata) {

            dim3 blocks((n + blockSize - 1) / blockSize);
            int size = 1 << ilog2ceil(n);

            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            cudaMalloc((void**) &dev_idata, n * sizeof(int));
            cudaMalloc((void**) &dev_odata, n * sizeof(int));
            cudaMalloc((void**) &dev_bools, n * sizeof(int));
            cudaMalloc((void**) &dev_indices, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_indices + n, 0, (size - n) * sizeof(int));

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanHelper(size, dev_indices);
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            int lastBool, lastIdx;
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIdx, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return lastBool + lastIdx;
        }

        __global__ void kernBitKNegative(int n, int *dev_idata, int *dev_bools, int bitK) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            dev_bools[index] = (dev_idata[index] & (1 << bitK)) != 0 ? 0 : 1;
        }

        __global__ void kernSplit(int n, int *dev_idata, int *dev_odata, int *dev_scan, int bitK, int totalFalses) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int data = dev_idata[index];
            int scanIdx = dev_scan[index];
            if ((data & (1 << bitK)) == 0) {
                dev_odata[scanIdx] = data;
            } else {
                dev_odata[index - scanIdx + totalFalses] = data;
            }
        }

        void radixSort(int n, int *odata, const int *idata) {

            dim3 blocks((n + blockSize - 1) / blockSize);
            int size = 1 << ilog2ceil(n);
            int maxNum = 0;
            for (int i = 0; i < n; ++i) {
                maxNum = std::max(maxNum, idata[i]);
            }
            int maxBit = ilog2ceil(maxNum);

            int *dev_data1, *dev_data2, *dev_scan;
            cudaMalloc((void**) &dev_data1, n * sizeof(int));
            cudaMalloc((void**) &dev_data2, n * sizeof(int));
            cudaMalloc((void**) &dev_scan, size * sizeof(int));

            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            for (int i = 0; i <= maxBit; ++i) {
                int lastBool, lastScan;
                kernBitKNegative<<<blocks, blockSize>>>(n, dev_data1, dev_scan, i);
                cudaMemcpy(&lastBool, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemset(dev_scan + n, 0, (size - n) * sizeof(int));
                scanHelper(size, dev_scan);
                cudaMemcpy(&lastScan, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                kernSplit<<<blocks, blockSize>>>(n, dev_data1, dev_data2, dev_scan, i, lastBool + lastScan);
                std::swap(dev_data1, dev_data2);
            }

            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data1);
            cudaFree(dev_data2);
            cudaFree(dev_scan);
        }
    }
}
