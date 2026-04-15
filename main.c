#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void cleanup_matrices(double *A, double *B, double *C) {
    free(A);
    free(B);
    free(C);
}

void free_local_blocks(double *Ablock, double *Bblock, double *Cblock) {
    free(Ablock);
    free(Bblock);
    free(Cblock);
}

bool allocate_local_blocks(double **Ablock, double **Bblock, double **Cblock, int rows, int n2, int cols) {
    *Ablock = malloc(rows * n2 * sizeof(double));
    if (!*Ablock) return false;
    
    *Bblock = malloc(n2 * cols * sizeof(double));
    if (!*Bblock) {
        free(*Ablock);
        return false;
    }
    
    *Cblock = calloc(rows * cols, sizeof(double));
    if (!*Cblock) {
        free(*Ablock);
        free(*Bblock);
        return false;
    }
    
    return true;
}

double valA(int i, int j) {
    return sin(i * 0.01) + cos(j * 0.02) + (i * j % 7) * 0.1;
}

double valB(int i, int j) {
    return cos(i * 0.03) + sin(j * 0.04) + (i + j % 5) * 0.5;
}

bool initialize_matrices(double **A, double **B, double **C, int n1, int n2, int n3) {
    *A = malloc(n1 * n2 * sizeof(double));
    if (!*A) return false;
    
    *B = malloc(n2 * n3 * sizeof(double));
    if (!*B) {
        free(*A);
        return false;
    }
    
    *C = calloc(n1 * n3, sizeof(double));
    if (!*C) {
        free(*A);
        free(*B);
        return false;
    }
    
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            (*A)[i * n2 + j] = valA(i, j);
        }
    }
    
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; j++) {
            (*B)[i * n3 + j] = valB(i, j);
        }
    }
    
    return true;
}

bool verify_result(double *C, double *A, double *B, int n1, int n2, int n3) {
    double *C_expected = calloc(n1 * n3, sizeof(double));
    if (!C_expected) return false;
    for (int i = 0; i < n1; i++) {
        for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n3; j++) {
                C_expected[i * n3 + j] += A[i * n2 + k] * B[k * n3 + j];
            }
        }
    }
    
    for (int i = 0; i < n1 * n3; i++) {
        if (C[i] != C_expected[i]) {
            free(C_expected);
            return false;
        }
    }
    free(C_expected);
    return true;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == 0) printf("Result: ERROR\n");
        MPI_Finalize();
        return 1;
    }
    
    int n1 = atoi(argv[1]);
    int n2 = atoi(argv[2]);
    int n3 = atoi(argv[3]);
    
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int px = dims[0], py = dims[1];
    
    int periods[2] = {0, 0};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    
    if (n1 % px != 0 || n3 % py != 0) {
        if (rank == 0) printf("Result: ERROR\n");
        MPI_Finalize();
        return 1;
    }
    
    int rows = n1 / px;
    int cols = n3 / py;
    
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid_comm, remain_dims, &row_comm);
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid_comm, remain_dims, &col_comm);
    
    double *A = NULL, *B = NULL, *C = NULL;
    MPI_Datatype col_type, res_type;
    
    if (rank == 0) {
        if (!initialize_matrices(&A, &B, &C, n1, n2, n3)) {
            printf("Result: ERROR\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Type_vector(n2, cols, n3, MPI_DOUBLE, &col_type);
        MPI_Type_create_resized(col_type, 0, cols * sizeof(double), &col_type);
        MPI_Type_commit(&col_type);   
        MPI_Type_vector(rows, cols, n3, MPI_DOUBLE, &res_type);
        MPI_Type_create_resized(res_type, 0, sizeof(double), &res_type);
        MPI_Type_commit(&res_type);
    }
    
    double *Ablock = NULL, *Bblock = NULL, *Cblock = NULL;
    if (!allocate_local_blocks(&Ablock, &Bblock, &Cblock, rows, n2, cols)) {
        if (rank == 0) printf("Result: ERROR\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double start = MPI_Wtime();
    if (coords[1] == 0) {
        MPI_Scatter(A, rows * n2, MPI_DOUBLE, Ablock,
                    rows * n2, MPI_DOUBLE, 0, col_comm);
    }
    MPI_Bcast(Ablock, rows * n2, MPI_DOUBLE, 0, row_comm);    
    if (coords[0] == 0) {
        MPI_Scatter(B, 1, col_type, Bblock,
                    n2 * cols, MPI_DOUBLE, 0, row_comm);
    }
    MPI_Bcast(Bblock, n2 * cols, MPI_DOUBLE, 0, col_comm);
    
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < n2; k++) {
            for (int j = 0; j < cols; j++) {
                Cblock[i * cols + j] += Ablock[i * n2 + k] * Bblock[k * cols + j];
            }
        }
    }
    double end = MPI_Wtime();
    double computation_time = end - start;
    
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int p = 0; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(grid_comm, p, 2, p_coords);
            sendcounts[p] = 1;
            displs[p] = (p_coords[0] * rows * n3) + (p_coords[1] * cols);
        }
    }
    
    MPI_Gatherv(Cblock, rows * cols, MPI_DOUBLE, C,
                sendcounts, displs, res_type, 0, grid_comm);
    
    if (rank == 0) {
        bool correct = verify_result(C, A, B, n1, n2, n3);
        printf("Matrix: %dx%dx%d\n", n1, n2, n3);
        printf("Time: %.6f seconds\n", computation_time);
        printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
        cleanup_matrices(A, B, C);
        free(sendcounts);
        free(displs);
        MPI_Type_free(&col_type);
        MPI_Type_free(&res_type);
    }
    
    free_local_blocks(Ablock, Bblock, Cblock);
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
