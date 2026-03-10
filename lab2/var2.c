#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define EPSILON 1e-12

double calculate_error(double* solution, double* exact, int N) {
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(solution[i] - exact[i]);
        if (diff > error) error = diff;
    }
    return error;
}

void conjugate_gradients_omp_v2(FullSystem* system, int N, int max_iterations) {
    double* r = malloc(N * sizeof(double));
    double* z = malloc(N * sizeof(double));
    double* full_part = malloc(N * sizeof(double));
    
    double stop_criterion;
    int converged = 0;
    int iteration = 0;
    double r_norm;
    double r_dot_r, Az_dot_z, alpha, beta;
    
    // Эти переменные выносим ДО parallel
    double local_b_norm = 0.0;
    double local_r_dot_r, local_Az_dot_z, local_r_norm_sq;
    
    #pragma omp parallel
    {
        // Ax0
        #pragma omp for
        for (int i = 0; i < N; i++) {
            full_part[i] = 0.0;
            for (int j = 0; j < N; j++) {
                full_part[i] += system->A[i * N + j] * system->x0[j];
            }
        }

        // r0 = b - A*x0
        #pragma omp for
        for (int i = 0; i < N; i++) {
            r[i] = system->b[i] - full_part[i];
        }
        
        // z0 = r0
        #pragma omp for
        for (int i = 0; i < N; i++) {
            z[i] = r[i];
        }
        
        local_b_norm = 0.0;
        #pragma omp for reduction(+:local_b_norm)
        for (int i = 0; i < N; i++) {
            local_b_norm += system->b[i] * system->b[i];
        }
        
        #pragma omp single
        {
            double b_norm = sqrt(local_b_norm);
            stop_criterion = EPSILON * b_norm;
        }
        
        do {
            // Az
            #pragma omp for
            for (int i = 0; i < N; i++) {
                full_part[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    full_part[i] += system->A[i * N + j] * z[j];
                }
            }

            local_r_dot_r = 0.0;
            local_Az_dot_z = 0.0;
            
            #pragma omp for reduction(+:local_r_dot_r, local_Az_dot_z)
            for (int i = 0; i < N; i++) {
                local_r_dot_r += r[i] * r[i];
                local_Az_dot_z += full_part[i] * z[i];
            }
            
            #pragma omp single
            {
                r_dot_r = local_r_dot_r;
                Az_dot_z = local_Az_dot_z;
                alpha = r_dot_r / Az_dot_z;
            }
            
            // x = x + alpha * z
            #pragma omp for
            for (int i = 0; i < N; i++) {
                system->x0[i] += alpha * z[i];
            }
            
            // r = r - alpha * A*z
            #pragma omp for
            for (int i = 0; i < N; i++) {
                r[i] -= alpha * full_part[i];
            }
            
            local_r_norm_sq = 0.0;
            #pragma omp for reduction(+:local_r_norm_sq)
            for (int i = 0; i < N; i++) {
                local_r_norm_sq += r[i] * r[i];
            }
            
            #pragma omp single
            {
                r_norm = sqrt(local_r_norm_sq);
                
                if (r_norm < stop_criterion) {
                    converged = 1;
                    printf("Iterations %d\n", iteration + 1);
                }
                
                beta = (r_norm * r_norm) / r_dot_r;
            }
            
            // z = r + beta * z
            #pragma omp for
            for (int i = 0; i < N; i++) {
                z[i] = r[i] + beta * z[i];
            }
            
            #pragma omp single
            {
                iteration++;
            }
            
        } while (iteration < max_iterations && !converged);
    }
    
    free(r);
    free(z);
    free(full_part);
}

int main(int argc, char** argv) {
    const int N = 10000;
    const int max_iterations = 1000;
    
    FullSystem* system = create_full_system(N);
    if (!system) {
        printf("Failed to create system");
        return 1;
    }
    
    double start_time, end_time;
    start_time = omp_get_wtime();
    conjugate_gradients_omp_v2(system, N, max_iterations);
    end_time = omp_get_wtime();
    
    double error = calculate_error(system->x0, system->u, N);
    printf("Time: %f seconds\n", end_time - start_time);
    printf("Error: %e\n", error);
    
    free_full_system(system);
    return 0;
}