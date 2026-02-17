#include <math.h>
#include "init_sys_1.h"

FullSystem* create_full_system(int N) {
    FullSystem* system = (FullSystem*)malloc(sizeof(FullSystem));
    if (!system) return NULL;
    
    system->N = N;
    system->A = (double*)malloc(N * N * sizeof(double));
    if (!system->A) {
        free(system);
        return NULL;
    }
    
    system->x0 = (double*)malloc(N * sizeof(double));
    if (!system->x0) {
        free(system->A);
        free(system);
        return NULL;
    }
    
    system->u = (double*)malloc(N * sizeof(double));
    if (!system->u) {
        free(system->A);
        free(system->x0);
        free(system);
        return NULL;
    }
    
    system->b = (double*)malloc(N * sizeof(double));
    if (!system->b) {
        free(system->A);
        free(system->x0);
        free(system->u);
        free(system);
        return NULL;
    }
    
    return system;
}

void free_full_system(FullSystem* system) {
    if (!system) return;
    free(system->A);
    free(system->x0);
    free(system->u);
    free(system->b);
    free(system);
}

LocalData* create_local_data(int rank, int size, int N) {
    LocalData* local = (LocalData*)malloc(sizeof(LocalData));
    if (!local) return NULL;

    local->rank = rank;
    local->size = size;
    
    int base_rows = N / size;
    int remainder = N % size;
    local->local_rows = base_rows + (rank < remainder ? 1 : 0);

    local->send_counts = (int*)malloc(size * sizeof(int));
    if (!local->send_counts) {
        free(local);
        return NULL;
    }
    
    local->displacements = (int*)malloc(size * sizeof(int));
    if (!local->displacements) {
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    local->matrix_counts = (int*)malloc(size * sizeof(int));
    if (!local->matrix_counts) {
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    local->matrix_displs = (int*)malloc(size * sizeof(int));
    if (!local->matrix_displs) {
        free(local->matrix_counts);
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    // Вычисляем смещения
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = base_rows + (i < remainder ? 1 : 0);
        local->send_counts[i] = rows;
        local->displacements[i] = offset;
        local->matrix_counts[i] = rows * N;
        local->matrix_displs[i] = offset * N;
        offset += rows;
    }
    
    // Память для данных
    local->local_A = (double*)malloc(local->local_rows * N * sizeof(double));
    if (!local->local_A) {
        free(local->matrix_counts);
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    local->full_x = (double*)malloc(N * sizeof(double));
    if (!local->full_x) {
        free(local->local_A);
        free(local->matrix_counts);
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    local->full_b = (double*)malloc(N * sizeof(double));
    if (!local->full_b) {
        free(local->full_x);
        free(local->local_A);
        free(local->matrix_counts);
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    local->full_u = (double*)malloc(N * sizeof(double));
    if (!local->full_u) {
        free(local->full_b);
        free(local->full_x);
        free(local->local_A);
        free(local->matrix_counts);
        free(local->displacements);
        free(local->send_counts);
        free(local);
        return NULL;
    }
    
    return local;
}

void free_local_data(LocalData* local) {
    if (!local) return;
    free(local->send_counts);
    free(local->displacements);
    free(local->matrix_counts);
    free(local->matrix_displs);
    free(local->local_A);
    free(local->full_x);
    free(local->full_b);
    free(local->full_u);
    free(local);
}

void full_A(double* A, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N + j] = 1.0 / (i + j + 1.0);
        }
    }
}

void full_x0(double* x0, int N) {
    for (int i = 0; i < N; ++i) {
        x0[i] = 0.0;
    }
}

void full_u(double* u, int N) {
    for (int i = 0; i < N; ++i) {
        u[i] = sin(2 * M_PI * i / N);
    }
}

void full_b(double* b, const double* u, const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            b[i] += A[i * N + j] * u[j];
        }
    }
}

