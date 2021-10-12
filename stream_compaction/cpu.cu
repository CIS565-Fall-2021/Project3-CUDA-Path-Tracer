#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <iostream>


namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scanNoTimer(int n, int* odata, const int* idata) {
            // exclusive
            odata[0] = 0;
            for (int k = 1; k < n; k++) {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanNoTimer(n, odata, idata);
            timer().endCpuTimer();
        }



        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int index = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[index] = idata[i];
                    index++;
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // compute temp array
            int* tempArr = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    tempArr[i] = 1;
                }
                else {
                    tempArr[i] = 0;
                }
            }
            
            // exclusive scan
            scanNoTimer(n, odata, tempArr);

            // scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (tempArr[i] != 0) {
                    odata[odata[i]] = idata[i];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }
    }
}
