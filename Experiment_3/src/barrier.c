// barrier.c - Demonstrates #pragma omp barrier phased synchronization
#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Phase 1
        printf("Thread %d before barrier\n", tid);

        #pragma omp barrier  // All threads wait here

        // Phase 2
        printf("Thread %d after barrier\n", tid);
    }

    return 0;
}
