#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

typedef struct {
    int N;         
    double* A;     
    double* x0;   
    double* u;
    double* b;     
} FullSystem;

FullSystem* create_full_system(int N);
void free_full_system(FullSystem* system);
#endif