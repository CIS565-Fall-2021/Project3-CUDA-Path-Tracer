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


        // only run for 1 thread at a time
        __global__ void kernUpdateArr(int idx, int val, int *arr) {
            arr[idx] = val;
        }

        __global__ void kernScanUpSweep(int n, int *data, int pow2) {
            /*int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (idx % (2 * pow2) == 0) {
                data[idx + 2 * pow2 - 1] += data[idx + pow2 - 1];
            }*/

            // optimized solution
            size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            idx = 2 * pow2 * (idx + 1) - 1;
            if (idx >= n) {
                return;
            }
            data[idx] += data[idx - pow2];
            
        }

        __global__ void kernScanDownSweep(int n, int *data, int pow2) {
            /*int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (idx % (2 * pow2) == 0) {
                int temp = data[idx + pow2 - 1];
                data[idx + pow2 - 1] = data[idx + 2 * pow2 - 1];
                data[idx + 2 * pow2 - 1] += temp;
            }*/
            // optimized solution
            size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            idx = 2 * pow2 * (idx + 1) - 1;
            if (idx >= n) {
                return;
            }
            int temp = data[idx - pow2];
            data[idx - pow2] = data[idx];
            data[idx] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_arr;
            int maxDepth = ilog2ceil(n);
            int size = pow(2, maxDepth);

            cudaMalloc((void**)&dev_arr, size * sizeof(int));
            cudaMemcpy(dev_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blockNumPow((size + blockSize - 1) / blockSize);


            timer().startGpuTimer();
            for (int d = 0; d < maxDepth; d++) {
                blockNumPow = (size / pow(2, d + 1) + blockSize - 1) / blockSize;
                kernScanUpSweep << <blockNumPow, blockSize >> > (size, dev_arr, pow(2, d));
            }

            kernUpdateArr << <1, 1 >> > (size - 1, 0, dev_arr);

            for (int d = maxDepth - 1; d >= 0; d--) {
                blockNumPow = (size / pow(2, d + 1) + blockSize - 1) / blockSize;
                kernScanDownSweep << <blockNumPow, blockSize >> > (size, dev_arr, pow(2, d));
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_arr);
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
            int* dev_idata, *dev_odata, *dev_bools, *dev_indices;
            int maxDepth = ilog2ceil(n);
            int size = pow(2, maxDepth);
   
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            cudaMalloc((void**)&dev_indices, size * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blockNum((n + blockSize - 1) / blockSize);
            dim3 blockNumPow((size + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            Common::kernMapToBoolean << <blockNum, blockSize >> > (size, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, size * sizeof(int), cudaMemcpyHostToDevice);

            // scan
            for (int d = 0; d < maxDepth; d++) {
                blockNumPow = (size / pow(2, d + 1) + blockSize - 1) / blockSize;
                kernScanUpSweep << <blockNumPow, blockSize >> > (size, dev_indices, pow(2, d));
            }

            kernUpdateArr << <1, 1 >> > (size - 1, 0, dev_indices);

            for (int d = maxDepth - 1; d >= 0; d--) {
                blockNumPow = (size / pow(2, d + 1) + blockSize - 1) / blockSize;
                kernScanDownSweep << <blockNumPow, blockSize >> > (size, dev_indices, pow(2, d));
            }

            // scatter
            Common::kernScatter << <blockNum, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int* bools = new int[n];
            cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i]) {
                    count++;
                }
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}
