#include <math.h>
#include "utils.h"

void fill_x0(double* x0, int N) {
    for (int i = 0; i < N; ++i) {
        x0[i] = 0.0;
    }
}

static void fill_u(double* u, int N) {
    for (int i = 0; i < N; ++i) {
        u[i] = sin(2 * M_PI * i / N);
    }
}

static void fill_b(double* b, const double* u, const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            b[i] += A[i * N + j] * u[j];
        }
    }
}

void fill_A(double* A, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N + j] = 1.0 / (i + j + 1.0);
        }
    }
}

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
    
    fill_A(system->A, N);
    fill_x0(system->x0, N);
    fill_u(system->u, N);
    fill_b(system->b, system->u, system->A, N);
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