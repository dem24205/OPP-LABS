#ifndef STRUCTURES_1_h
#define STRUCTURES_1_h

#include <stdlib.h>

typedef struct {
    int N;         
    double* A;     
    double* x0;   
    double* u;
    double* b;     
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
    double* full_x;       
    double* full_b;       
    double* full_u;       
} LocalData;

FullSystem* create_full_system(int N);
void free_full_system(FullSystem* system);

LocalData* create_local_data(int rank, int size, int N);
void free_local_data(LocalData* local);
#endif