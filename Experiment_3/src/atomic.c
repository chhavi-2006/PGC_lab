// atomic.c - Uses #pragma omp atomic for lightweight hardware-level synchronization
#include <stdio.h>
#include <omp.h>

#define NUM_INCREMENTS 10000000

int main() {
    long long counter = 0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int chunk = NUM_INCREMENTS / nthreads;
        int extra = (tid < NUM_INCREMENTS % nthreads) ? 1 : 0;

        for (int i = 0; i < chunk + extra; i++) {
            #pragma omp atomic
            counter++;
        }
    }

    printf("Counter=%lld\n", counter);
    return 0;
}
