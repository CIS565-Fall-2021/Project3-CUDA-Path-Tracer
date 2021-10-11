#pragma once
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();


        void PerformThrustSort(int n, int* odata, const int* idata);
        void PerformGPUSort(int n, int* odata, const int* idata);
    }
}