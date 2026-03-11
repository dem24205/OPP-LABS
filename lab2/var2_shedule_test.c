#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define EPSILON 1e-10
#define ITERATION_MAX_NUM 1000

double calculate_error(double* solution, double* exact, int N) {
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(solution[i] - exact[i]);
        if (diff > error) error = diff;
    }
    return error;
}

void conjugate_gradients(FullSystem* system, int N) {
    double* r = (double*)malloc(N * sizeof(double));
    if (!r) {
        printf("Failed to allocate memory for r\n");
        return;
    }

    double* z = (double*)malloc(N * sizeof(double));
    if (!z) {
        printf("Failed to allocate memory for z\n");
        free(r);
        return;
    }

    double* full_part = (double*)malloc(N * sizeof(double));
    if (!full_part) {
        printf("Failed to allocate memory for full_part\n");
        free(r);
        free(z);
        return;
    }

    double stop_criterion;
    int converged = 0;
    int iteration = 0;
    double r_norm = 0.0;
    double r_dot_r = 0.0;
    double Az_dot_z = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    #pragma omp parallel
    {
        // Ax0
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            full_part[i] = 0.0;
            for (int j = 0; j < N; j++) {
                full_part[i] += system->A[i * N + j] * system->x0[j];
            }
        }

        // r0 = b - A*x0
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            r[i] = system->b[i] - full_part[i];
        }

        // z0 = r0
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            z[i] = r[i];
        }

        #pragma omp for reduction(+:r_dot_r) schedule(runtime)
        for (int i = 0; i < N; i++) {
            r_dot_r += system->b[i] * system->b[i];
        }

        #pragma omp single
        {
            stop_criterion = EPSILON * sqrt(r_dot_r);
            r_dot_r = 0.0;
        }

        do {
            // Az
            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                full_part[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    full_part[i] += system->A[i * N + j] * z[j];
                }
            }

            #pragma omp for reduction(+:r_dot_r, Az_dot_z) schedule(runtime)
            for (int i = 0; i < N; i++) {
                r_dot_r += r[i] * r[i];
                Az_dot_z += full_part[i] * z[i];
            }

            #pragma omp single
            {
                alpha = r_dot_r / Az_dot_z;
            }

            // x = x + alpha * z
            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                system->x0[i] += alpha * z[i];
            }

            // r = r - alpha * A*z
            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                r[i] -= alpha * full_part[i];
            }

            #pragma omp for reduction(+:r_norm) schedule(runtime)
            for (int i = 0; i < N; i++) {
                r_norm += r[i] * r[i];
            }

            #pragma omp single
            {
                r_norm = sqrt(r_norm);

                if (r_norm < stop_criterion) {
                    converged = 1;
                }

                beta = (r_norm * r_norm) / r_dot_r;
                r_dot_r = 0.0;
                Az_dot_z = 0.0;
                r_norm = 0.0;
                iteration++;
            }

            if (converged) break;

            // z = r + beta * z
            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                z[i] = r[i] + beta * z[i];
            }

        } while (iteration < ITERATION_MAX_NUM);
    }

    free(r);
    free(z);
    free(full_part);
}

int main(int argc, char** argv) {
    const int N = 10000;
    FullSystem* system = create_full_system(N);
    if (!system) {
        printf("Failed to create system");
        return 1;
    }
    double start_time, end_time;
    start_time = omp_get_wtime();
    conjugate_gradients(system, N);
    end_time = omp_get_wtime();
    double error = calculate_error(system->x0, system->u, N);
    printf("Time: %f seconds\n", end_time - start_time);
    printf("Error: %e\n", error);
    free_full_system(system);
    return 0;
}


