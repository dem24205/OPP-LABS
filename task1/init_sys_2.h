#ifndef INITSYS2_H
#define INITSYS2_H

#include <stdlib.h>

typedef struct {
    int N;         
    double* A;     
    double* x0;   
    double* u;   
} FullSystem;

typedef struct {
    int rank;           
    int size;          
    int local_rows;

    int* send_counts;      
    int* displacements;    
    int* matrix_counts;  
    int* matrix_displs;

    double* local_A;  
    double* local_x; 
    double* local_b;    
    double* full_u;
} LocalData;

FullSystem* create_full_system(int N);
void free_full_system(FullSystem* system);

LocalData* create_local_data(int rank, int size, int N);
void free_local_data(LocalData* local);

void full_A(double* A, int N);
void full_x0(double* x0, int N);
void full_u(double* u, int N);

#endif // STRUCTURES_H