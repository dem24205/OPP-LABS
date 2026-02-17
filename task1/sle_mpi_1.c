#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "init_sys_1.h"

#define EPSILON 1e-10

void distribute_data(FullSystem* full_system, LocalData* local, int N, MPI_Comm comm) {
    int rank = local->rank;
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            local->full_u[i] = full_system->u[i];
            local->full_b[i] = full_system->b[i];
            local->full_x[i] = full_system->x0[i];
        }
    }
    MPI_Bcast(local->full_u, N, MPI_DOUBLE, 0, comm);
    MPI_Bcast(local->full_b, N, MPI_DOUBLE, 0, comm);
    MPI_Bcast(local->full_x, N, MPI_DOUBLE, 0, comm);
    
    MPI_Scatterv(full_system ? full_system->A : NULL,
                 local->matrix_counts, local->matrix_displs, MPI_DOUBLE,
                 local->local_A, local->local_rows * N, MPI_DOUBLE,
                 0, comm);
}

void matvec_local(LocalData* local, double* full_x, double* local_result, int N) {
    for (int i = 0; i < local->local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_result[i] += local->local_A[i * N + j] * full_x[j];
        }
    }
}

double dot_product_full(double* v1, double* v2, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

//МЕТОД СОПРЯЖЕННЫХ ГРАДИЕНТОВ
void conjugate_gradients(LocalData* local, int N, int max_iterations) {
    int rank = local->rank;
    double* r = malloc(N * sizeof(double));
    if (!r) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double* z = malloc(N * sizeof(double));
    if (!z) {
        free(r);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double* local_part = malloc(local->local_rows * sizeof(double));
    if (!local_part) {
        free(r);
        free(z);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double* full_part = malloc(N * sizeof(double));
    if (!full_part) {
        free(r);
        free(z);
        free(local_part);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    matvec_local(local, local->full_x, local_part, N);
    MPI_Allgatherv(local_part, local->local_rows, MPI_DOUBLE,
                   full_part, local->send_counts, local->displacements,
                   MPI_DOUBLE, MPI_COMM_WORLD);
    //r0 = b - A*x0
    for (int i = 0; i < N; i++) {
        r[i] = local->full_b[i] - full_part[i];
    }
    
    //z0 = r0 (отдельно, для ясности)
    for (int i = 0; i < N; i++) {
        z[i] = r[i];
    }
    
    double b_norm = sqrt(dot_product_full(local->full_b, local->full_b, N));
    double stop_criterion = EPSILON * b_norm;
    int iteration = 0;
    double r_norm;
    
    do {
        //A*z
        matvec_local(local, z, local_part, N);

        MPI_Allgatherv(local_part, local->local_rows, MPI_DOUBLE,
                       full_part, local->send_counts, local->displacements,
                       MPI_DOUBLE, MPI_COMM_WORLD);
        
        //alpha = (r,r) / (A*z, z)
        double r_dot_r = dot_product_full(r, r, N);
        double Az_dot_z = dot_product_full(full_part, z, N);
        double alpha = r_dot_r / Az_dot_z;
        
        //x = x + alpha * z
        for (int i = 0; i < N; i++) {
            local->full_x[i] += alpha * z[i];
        }
        
        //r = r - alpha * A*z
        for (int i = 0; i < N; i++) {
            r[i] -= alpha * full_part[i];
        }
        
        r_norm = sqrt(dot_product_full(r, r, N));
        
        if (r_norm < stop_criterion) {
            if (rank == 0) {
                printf("Converged after %d iterations, residual = %e\n", 
                       iteration + 1, r_norm);
            }
            break;
        }
        
        //beta = (r_new, r_new) / (r_old, r_old)
        double beta = (r_norm * r_norm) / r_dot_r;
        
        //z = r + beta * z
        for (int i = 0; i < N; i++) {
            z[i] = r[i] + beta * z[i];
        }
        iteration++;
    } while (iteration < max_iterations);    

    free(r);
    free(z);
    free(local_part);
    free(full_part);
}

double calculate_error(double* solution, double* exact, int N) {
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(solution[i] - exact[i]);
        if (diff > error) error = diff;
    }
    return error;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    start_time = MPI_Wtime(); 

    int N = 10000;
    int max_iterations = 1000;
    FullSystem* system = NULL;

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        system = create_full_system(N);
        if (!system) {
            printf("Process 0: failed to create full system\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        full_A(system->A, N);
        full_x0(system->x0, N);
        full_u(system->u, N);
        full_b(system->b, system->u, system->A, system->N);
    }
    
    LocalData* local = create_local_data(rank, size, N);
    if (!local) {
        printf("Process %d: failed to create local data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    distribute_data(system, local, N, MPI_COMM_WORLD);
    conjugate_gradients(local, N, max_iterations);
    end_time = MPI_Wtime();

    if (rank == 0) {
        double error = calculate_error(local->full_x, system->u, N);
        printf("\nMaximum error compared to exact solution: %e\n", error);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }

    free_local_data(local);
    if (rank == 0) {
        free_full_system(system);
    }
    
    MPI_Finalize();
    return 0;
}