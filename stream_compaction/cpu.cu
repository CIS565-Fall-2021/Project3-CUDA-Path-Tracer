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
        void scan(int n, int *odata, const int *idata) {
          
            timer().startCpuTimer();
            // TODO

            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i-1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int outIndex=0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[outIndex] = idata[i];
                    outIndex++;
                }
            }

            timer().endCpuTimer();
            return outIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

            int *arr_b = new int[n];
            int *arr_c = new int[n]; // exclusive array
            timer().startCpuTimer();


            int *ans = new int[n];

            int finalIndex = 0;
#pragma region NonPower2

            int *arr_z = new int[n];

            int nearesttwo = 1 << ilog2ceil(n);


            int difference = nearesttwo - n;

            for (int i = 0; i < difference; i++)
            {
                arr_z[i] = 0;
            }


            for (int i = 0; i < n; i++)
            {
                arr_z[i + difference] = idata[i];
            }

            n = n + difference;

            for (int i = 0; i < n; i++)
            {
                if (arr_z[i] == 0)
                {
                    arr_b[i] = 0;
                    continue;
                }
                arr_b[i] = 1;
            }

            arr_c[0] = 0;
            for (int i = 1; i < n; i++)
            {
                arr_c[i] = arr_b[i - 1] + arr_c[i - 1];
            }

            for (int i = 0; i < n; i++)
            {
                if (arr_b[i] == 0)
                {
                    continue;
                }
                int index = arr_c[i];
                odata[index] = idata[i];
                ans[index] = idata[i];
                finalIndex = index;
            }



#pragma endregion
            timer().endCpuTimer();
            return finalIndex+1;
        }
    }
}
