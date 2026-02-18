#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "init_sys_2.h"

#define EPSILON 1e-10

void distribute_data(FullSystem* full_system, LocalData* local, int N, MPI_Comm comm) {
    int rank = local->rank;
    //u - вектор, инициализированный на нулевом процессе
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            local->full_u[i] = full_system->u[i];
        }
    }
    //передается для инициализации кусков b для каждого процесса
    MPI_Bcast(local->full_u, N, MPI_DOUBLE, 0, comm);
   
    MPI_Scatterv(full_system ? full_system->A : NULL,
                 local->matrix_counts, local->matrix_displs, MPI_DOUBLE,
                 local->local_A, local->local_rows * N, MPI_DOUBLE,
                 0, comm);
   
    MPI_Scatterv(full_system ? full_system->x0 : NULL,
                 local->send_counts, local->displacements, MPI_DOUBLE,
                 local->local_x, local->local_rows, MPI_DOUBLE,
                 0, comm);
   
    for (int i = 0; i < local->local_rows; i++) {
        local->local_b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            local->local_b[i] += local->local_A[i * N + j] * local->full_u[j];
        }
    }
}

void matvec_distributed(LocalData* local,
                        double* local_x,
                        double* local_result,
                        int N) {
    int rank = local->rank;
    int size = local->size;

    //находим максимальный размер куска
    int max_rows = local->local_rows;
    MPI_Allreduce(&local->local_rows, &max_rows, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    //буферы под максимальный размер
    double* current_x = calloc(max_rows, sizeof(double)); //остаются нулями 
    if (!current_x) MPI_Abort(MPI_COMM_WORLD, 1);
    double* temp_x = calloc(max_rows, sizeof(double));
    if (!temp_x) {
        free(current_x);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int j = 0; j < local->local_rows; j++) {
        current_x[j] = local_x[j];
    }

    //свой вклад
    for (int i = 0; i < local->local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local->local_rows; j++) {
            int global_j = local->displacements[rank] + j;
            local_result[i] += local->local_A[i * N + global_j] * current_x[j];
        }
    }

    //кольцо
    for (int step = 1; step < size; step++) {
        int dest = (rank + 1) % size;
        int comm_src = (rank - 1 + size) % size;
        MPI_Sendrecv(current_x, max_rows, MPI_DOUBLE, dest,     17,
                     temp_x,    max_rows, MPI_DOUBLE, comm_src, 17,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //чей кусок мы получили
        int current_src = (rank - step + size) % size;
        //сколько реально значимых элементов пришло
        int src_count = local->send_counts[current_src];

        for (int i = 0; i < local->local_rows; i++) {
            for (int j = 0; j < src_count; j++) {
                int global_j = local->displacements[current_src] + j;
                local_result[i] += local->local_A[i * N + global_j] * temp_x[j];
            }
        }

        for (int j = 0; j < max_rows; j++) {
            current_x[j] = temp_x[j];
        }
    }
    free(temp_x);
    free(current_x);
}

//каждый процесс вычисляет свою часть суммы
//MPI_Allreduce собирает все частичные суммы и возвращает результат их суммирования для каждого процесса
double dot_product_local(LocalData* local, double* v1, double* v2) {
    double local_sum = 0.0;
    for (int i = 0; i < local->local_rows; i++) {
        local_sum += v1[i] * v2[i];
    }
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

//время выполнения
//количество итераций
void conjugate_gradients(LocalData* local, int N, int max_iterations) {
    int rank = local->rank;
    int size = local->size;double* local_r = malloc(local->local_rows * sizeof(double));
    if (!local_r) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double* local_z = malloc(local->local_rows * sizeof(double));
    if (!local_z) {
        free(local_r);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double* local_Az = malloc(local->local_rows * sizeof(double));
    if (!local_Az) {
        free(local_r);
        free(local_z);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
           
    matvec_distributed(local, local->local_x, local_Az, N);
    // r0 = b - A * x0
    for (int i = 0; i < local->local_rows; i++) {
        local_r[i] = local->local_b[i] - local_Az[i];
    }
   
    //z0 = r0
    for (int i = 0; i < local->local_rows; i++) {
        local_z[i] = local_r[i];
    }
    //⠵ < ε * ⠟
    double b_norm = sqrt(dot_product_local(local, local->local_b, local->local_b));
    double stop_criterion = EPSILON * b_norm;
    int iteration = 0;
    double r_norm;
    do {
        //Az = A * z_n
        matvec_distributed(local, local_z, local_Az, N);
        //alpha_{n+1} = (r_n, r_n) / (A z_n, z_n)
        double r_dot_r = dot_product_local(local, local_r, local_r); // (r_n, r_n)
        double Az_dot_z = dot_product_local(local, local_Az, local_z); // (A z_n, z_n)
        double alpha = r_dot_r / Az_dot_z;
       
        //x_{n+1} = x_n + alpha_{n+1} * z_n
        for (int i = 0; i < local->local_rows; i++) {
            local->local_x[i] += alpha * local_z[i];
        }
       
        //r_{n+1} = r_n - alpha_{n+1} * A z_n
        for (int i = 0; i < local->local_rows; i++) {
            local_r[i] -= alpha * local_Az[i];
        }
       
        r_norm = sqrt(dot_product_local(local, local_r, local_r));
       
        if (r_norm < stop_criterion) {
            if (rank == 0) {
                printf("Converged after %d iterations\n",
                       iteration+1);
            }
            break;
        }
       
        //beta_{n+1} = (r_{n+1}, r_{n+1}) / (r_n, r_n)
        double r_norm_dot_r_norm = r_norm * r_norm;
        double beta = r_norm_dot_r_norm / r_dot_r; //beta_{n+1}
       
        //z_{n+1} = r_{n+1} + beta_{n+1} * z_n
        for (int i = 0; i < local->local_rows; i++) {
            local_z[i] = local_r[i] + beta * local_z[i];
        }
        iteration++;
    } while (iteration < max_iterations);
    free(local_r);
    free(local_z);
    free(local_Az);
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
    }
   
    LocalData* local = create_local_data(rank, size, N);
    if (!local) {
        printf("Process %d: failed to create local data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
       
    distribute_data(system, local, N, MPI_COMM_WORLD);
    conjugate_gradients(local, N, max_iterations);
    end_time = MPI_Wtime();
    double* full_solution = malloc(N * sizeof(double));
    if (!full_solution) {
        printf("Process %d: failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Gatherv(local->local_x,
        local->local_rows, MPI_DOUBLE, full_solution, local->send_counts,
        local->displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
	double error = calculate_error(full_solution, system->u, N);
        printf("\nMaximum error compared to exact solution: %e\n", error);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }
    free(full_solution);
    free_local_data(local);
    if (rank == 0) {
        free_full_system(system);
    }
    MPI_Finalize();
    return 0;
}