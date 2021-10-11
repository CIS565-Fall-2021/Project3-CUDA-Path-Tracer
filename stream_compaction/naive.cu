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

        __global__ void kernScanStep(int n, const int *dev_in, int *dev_out, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int data = dev_in[index];
            if (index >= offset) {
                dev_out[index] = data + dev_in[index - offset];
            } else {
                dev_out[index] = data;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            dim3 blocks((n + blockSize - 1) / blockSize);
            int steps = ilog2ceil(n);
            int offset = 1;

            int *dev_temp1, *dev_temp2;
            cudaMalloc((void**) &dev_temp1, n * sizeof(int));
            cudaMalloc((void**) &dev_temp2, n * sizeof(int));

            cudaMemcpy(dev_temp1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 0; i < steps; ++i) {
                kernScanStep<<<blocks, blockSize>>>(n, dev_temp1, dev_temp2, offset);
                std::swap(dev_temp1, dev_temp2);
                offset *= 2;
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_temp1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_temp1);
            cudaFree(dev_temp2);
        }
    }
}
