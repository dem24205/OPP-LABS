#include <iostream>
#include <mpi.h>
#include <cstring>
#include <algorithm>

using namespace std;

int count_neighbors(char* grid, int x, int y, int rows, int cols) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + rows) % rows;
            int ny = (y + dy + cols) % cols;
            count += grid[nx * cols + ny];
        }
    }
    return count;
}

bool is_alive_cell(bool current, int neighbors) {
    return current ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
}

bool compare_grid_section(char* a, char* b, int local_rows, int cols) {
    for (int i = 0; i < local_rows; i++) {
        if (memcmp(a + (i+1)*cols, b + (i+1)*cols, cols * sizeof(char)) != 0) return false;
    }
    return true;
}

void compute_flags(char* current, char** history, int hist_len,
                  unsigned char* flags, int local_rows, int cols) {
    for (int i = 0; i < hist_len; ++i) {
        flags[i] = compare_grid_section(current, history[i], local_rows, cols) ? 1 : 0;
    }
}

void init_glider(char* grid, int X, int Y) {
    memset(grid, 0, X * Y * sizeof(char));
    grid[0*Y + 1] = 1;
    grid[1*Y + 2] = 1;
    grid[2*Y + 0] = 1;
    grid[2*Y + 1] = 1;
    grid[2*Y + 2] = 1;
}

bool check_global_stop(unsigned char* all_flags, int size, int hist_len) {
    for (int iter = 0; iter < hist_len; ++iter) {
        bool all_one = true;
        for (int p = 0; p < size; ++p) {
            if (all_flags[p * hist_len + iter] == 0) {
                all_one = false;
                break;
            }
        }
        if (all_one) return true;
    }
    return false;
}

bool compute_generation(char* local_grid, char* new_grid, char** history, int& history_count,
    int rank, int size, int local_rows, int Y, int max_iter, int cur_iter) {

    int extended_rows = local_rows + 2;
    MPI_Request req_send_up = MPI_REQUEST_NULL;
    MPI_Request req_send_down = MPI_REQUEST_NULL;
    MPI_Request req_recv_up = MPI_REQUEST_NULL;
    MPI_Request req_recv_down = MPI_REQUEST_NULL;
    MPI_Request req_alltoall = MPI_REQUEST_NULL;

    int prev_rank = (rank - 1 + size) % size;
    int next_rank = (rank + 1) % size;

    //1. отправка первой строки предыдущему ядру
    MPI_Isend(local_grid + 1*Y, Y, MPI_CHAR, prev_rank, 0, 
    MPI_COMM_WORLD, &req_send_up);

    //2. отправка последней строки последующему ядру
    MPI_Isend(local_grid + local_rows*Y, Y, MPI_CHAR, next_rank, 1, 
    MPI_COMM_WORLD, &req_send_down);

    //3. получение от предыдущего ядра его последней строки
    MPI_Irecv(local_grid + 0*Y, Y, MPI_CHAR, prev_rank, 1, 
    MPI_COMM_WORLD, &req_recv_up);

    //4. получение от последующего ядра его первой строки
    MPI_Irecv(local_grid + (local_rows+1)*Y, Y, MPI_CHAR, next_rank, 0, 
    MPI_COMM_WORLD, &req_recv_down);

    //5. вектор флагов останова
    unsigned char* flags = new unsigned char[history_count];
    compute_flags(local_grid, history, history_count, flags, local_rows, Y);

    if (cur_iter < max_iter - 1) {
        history[history_count] = new char[extended_rows * Y];
        memcpy(history[history_count], local_grid, extended_rows * Y * sizeof(char));
        history_count++;
    }

    //6. обмен векторами флагов останова со всеми ядрами
    int max_history_count;
    MPI_Allreduce(&history_count, &max_history_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    unsigned char* send_buf = new unsigned char[size * max_history_count]();
    unsigned char* recv_buf = new unsigned char[size * max_history_count]();

    for (int p = 0; p < size; p++) {
        for (int i = 0; i < history_count - 1; i++) {
            send_buf[p * max_history_count + i] = flags[i];
        }
    }

    MPI_Ialltoall(send_buf, max_history_count, MPI_UNSIGNED_CHAR,
        recv_buf, max_history_count, MPI_UNSIGNED_CHAR,
        MPI_COMM_WORLD, &req_alltoall);

    delete[] flags;

    //7. вычисление состояния клеток в строках кроме первой и последней
    for (int i = 2; i < local_rows; ++i) {
        for (int j = 0; j < Y; ++j) {
            int nb = count_neighbors(local_grid, i, j, extended_rows, Y);
            new_grid[i*Y + j] = is_alive_cell(local_grid[i*Y + j], nb);
        }
    }

    //8. освобождение буфера отправки первой строки предыдущему ядру
    MPI_Wait(&req_send_up, MPI_STATUS_IGNORE);

    //9. получение от предыдущего ядра его последней строки
    MPI_Wait(&req_recv_up, MPI_STATUS_IGNORE);

    //10. вычисление состояния клеток в первой строке
    for (int j = 0; j < Y; ++j) {
        int nb = count_neighbors(local_grid, 1, j, extended_rows, Y);
        new_grid[1*Y + j] = is_alive_cell(local_grid[1*Y + j], nb);
    }

    //11. освобождение буфера отправки последней строки последующему ядру
    MPI_Wait(&req_send_down, MPI_STATUS_IGNORE);

    //12. получение от последующего ядра его первой строки
    MPI_Wait(&req_recv_down, MPI_STATUS_IGNORE);

    //13. вычисление состояния клеток в последней строке
    for (int j = 0; j < Y; ++j) {
        int nb = count_neighbors(local_grid, local_rows, j, extended_rows, Y);
        new_grid[local_rows*Y + j] = is_alive_cell(local_grid[local_rows*Y + j], nb);
    }

    memcpy(local_grid, new_grid, extended_rows * Y * sizeof(char));

    //14. завершение обмена векторами флагов останова со всеми ядрами
    MPI_Wait(&req_alltoall, MPI_STATUS_IGNORE);

    //15. сравнение векторов флагов останова, полученные от всех ядер.
    bool stop = check_global_stop(recv_buf, size, history_count - 1);

    delete[] send_buf;
    delete[] recv_buf;

    return stop;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int X = 600, Y = 600;
    const int max_iter = 5000;
    
    int base_rows = X / size;
    int extra = X % size;
    int local_rows = base_rows + (rank < extra ? 1 : 0);
    int start_row = rank * base_rows + min(rank, extra);
    
    int dop_rows = local_rows + 2;
    char* local_grid = new char[dop_rows * Y]();
    char* new_grid = new char[dop_rows * Y]();
    
    char** history = new char*[max_iter];
    int history_count = 0;
    
    int* send_counts = nullptr;
    int* displs = nullptr;
    char* global = nullptr;
    
    if (rank == 0) {
        global = new char[X * Y]();
        init_glider(global, X, Y);
        
        send_counts = new int[size];
        displs = new int[size];
        
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int p_rows = base_rows + (p < extra ? 1 : 0);
            send_counts[p] = p_rows * Y;
            displs[p] = offset;
            offset += send_counts[p];
        }
    }
    
    MPI_Scatterv(global, send_counts, displs, MPI_CHAR,
                 local_grid + 1*Y, local_rows * Y, MPI_CHAR,
                 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        delete[] global;
        delete[] send_counts;
        delete[] displs;
    }
    
    history[0] = new char[dop_rows * Y];
    memcpy(history[0], local_grid, dop_rows * Y);
    history_count = 0;
    
    bool stop = false;
    int iter;

    double start_time = MPI_Wtime();
    for (iter = 0; iter < max_iter - 1 && !stop; ++iter) {
        stop = compute_generation(local_grid, new_grid, history, history_count,
                        rank, size, local_rows, Y, max_iter, iter);
    }
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        if (stop){
            cout << "Остановка на итерации " << iter << " (повтор состояния)\n";
        }
        else{
            cout << "Достигнут лимит итераций (" << max_iter << ")\n";
        }
        cout << "Время выполнения: " << end_time - start_time << " секунд" << endl;
    }
    
    for (int i = 0; i < history_count; ++i){
        delete[] history[i];
    }
    delete[] history;
    delete[] local_grid;
    delete[] new_grid;
    
    MPI_Finalize();
    return 0;
}
