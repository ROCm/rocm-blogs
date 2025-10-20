#include <hip/hip_runtime.h>
#include <stdio.h>

/**
 * Example HIP program that counts your GPUs, and checks that HIP can access each device.
 * Compile example: hipcc hello_hip.cpp -o hello_hip
 * Run example: LD_LIBRARY_PATH=.rockenv/lib/python3.12/site-packages/_rocm_sdk_core/lib/ ./hello_hip
 */
int main(void)
{
    int count;
    hipError_t err;

    err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        printf("hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("HIP found %d devices\n", count);

    for (int i = 0; i < count; ++i) {
        err = hipSetDevice(i);
        if (err != hipSuccess) {
            printf("hipSetDevice(%d) failed: %s\n", i, hipGetErrorString(err));
            continue;
        }

        int current;
        err = hipGetDevice(&current);  // confirm which device is active
        if (err != hipSuccess) {
            printf("hipGetDevice(%d) failed: %s\n", i, hipGetErrorString(err));
            continue;
        }
        printf("HIP found device %d\n", current);
    }

    return 0;
}

