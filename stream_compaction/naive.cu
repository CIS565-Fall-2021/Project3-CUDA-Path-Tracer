#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"




namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernNaiveScan(int n, int *odata, int *idata, int offset) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
                return;
            }
            if (idx >= offset) {
                odata[idx] = idata[idx - offset] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_arr1,* dev_arr2;

            cudaMalloc((void**)&dev_arr1, n * sizeof(int));
            cudaMalloc((void**)&dev_arr2, n * sizeof(int));
            cudaMemcpy(dev_arr1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blockNum((n + blockSize - 1) / blockSize);

            int maxDepth = ilog2ceil(n);
            timer().startGpuTimer();
            for (int d = 1; d <= maxDepth; d++) {
                kernNaiveScan << <blockNum, blockSize >> > (n, dev_arr2, dev_arr1, pow(2.0,d-1));

                // ping pong
                if (d < maxDepth) {
                    int* temp = dev_arr1;
                    dev_arr1 = dev_arr2;
                    dev_arr2 = temp;
                }
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_arr2, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            cudaFree(dev_arr1);
            cudaFree(dev_arr2);

        }
    }
}
