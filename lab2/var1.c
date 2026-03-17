#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define EPSILON 1e-12
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
    double* r = malloc(N * sizeof(double));
    if (!r) {
        printf("Failed to allocate r vector\n");
        return;
    }
    
    double* z = malloc(N * sizeof(double));
    if (!z) {
        printf("Failed to allocate z vector\n");
        free(r);
        return;
    }
    
    double* full_part = malloc(N * sizeof(double));
    if (!full_part) {
        printf("Failed to allocate full_part\n");
        free(r);
        free(z);
        return;
    }
    
    //Ax0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        full_part[i] = 0.0;
        for (int j = 0; j < N; j++) {
            full_part[i] += system->A[i * N + j] * system->x0[j];
        }
    }

    //r0 = b - A*x0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        r[i] = system->b[i] - full_part[i];
    }
    
    //z0 = r0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        z[i] = r[i];
    }
    
    double b_norm = 0.0;
    #pragma omp parallel for reduction(+:b_norm)
    for (int i = 0; i < N; i++) {
        b_norm += system->b[i] * system->b[i];
    }
    
    double stop_criterion = EPSILON * sqrt(b_norm);;
    int iteration = 0;
    double r_norm = 0.0;
    
    do {
        //Az
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            full_part[i] = 0.0;
            for (int j = 0; j < N; j++) {
                full_part[i] += system->A[i * N + j] * z[j];
            }
        }

        //alpha = (r,r) / (A*z, z)
        double r_dot_r = 0.0;
        double Az_dot_z = 0.0;
        #pragma omp parallel for reduction(+:r_dot_r, Az_dot_z)
        for (int i = 0; i < N; i++) {
            r_dot_r += r[i] * r[i];
            Az_dot_z += full_part[i] * z[i];
        }
        double alpha = r_dot_r / Az_dot_z;
        
        //x = x + alpha * z
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            system->x0[i] += alpha * z[i];
        }
        
        //r = r - alpha * A*z
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            r[i] -= alpha * full_part[i];
        }
        
        #pragma omp parallel for reduction(+:r_norm)
        for (int i = 0; i < N; i++) {
            r_norm += r[i] * r[i];
        }
        r_norm = sqrt(r_norm);
        
        if (r_norm < stop_criterion) break;
        
        //beta = (r_new, r_new) / (r_old, r_old)
        double beta = (r_norm * r_norm) / r_dot_r;
        r_norm = 0.0;

        //z = r + beta * z
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            z[i] = r[i] + beta * z[i];
        }
        
        iteration++;
    } while (iteration < ITERATION_MAX_NUM);
    
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
