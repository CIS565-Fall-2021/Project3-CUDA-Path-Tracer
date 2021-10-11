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

        void scanHelper(int n, int* odata, const int* idata) {
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanHelper(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int j = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[j++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *boolArray = new int[n];
            int *scanArray = new int[n];
            for (int i = 0; i < n; ++i) {
                boolArray[i] = idata[i] == 0 ? 0 : 1;
            }
            scanHelper(n, scanArray, boolArray);
            int num = 0;
            for (int i = 0; i < n; ++i) {
                if (boolArray[i] != 0) {
                    odata[scanArray[i]] = idata[i];
                    ++num;
                }
            }
            timer().endCpuTimer();
            return num;
        }
    }
}
