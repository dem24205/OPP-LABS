#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>

class DistributedTaskBalancer {
public:
    DistributedTaskBalancer(int my_rank, int total_ranks);
    ~DistributedTaskBalancer();

    void execute(int num_iterations);

private:
    //теги для MPI-сообщений
    static constexpr int REQUEST_TAG  = 100; //запрос задач 
    static constexpr int RESPONSE_TAG = 101; //ответ с количеством задач
    static constexpr int DATA_TAG     = 102; //сами задачи

    static constexpr int MIN_SHARE = 4;
    
    static constexpr long long TOTAL_TASKS = 400000LL;
    static constexpr long long TOTAL_WEIGHT = TOTAL_TASKS * 1000LL;
    
    //флаг для балансировки
    static constexpr bool ENABLE_BALANCING = true;  
    int rank;
    int world_size;

    //очередь задач, где каждая задача это ее вес
    std::vector<int> local_tasks;
    long long pending = 0; //остаток
    long long completed = 0; //выполнено

    std::mutex mtx;
    std::atomic<bool> running{true};
    std::thread responder_thread;

    void compute_task(int weight);
    void response_handler();
    bool try_steal_work();

    int calculate_task_weight(int iter);
};

DistributedTaskBalancer::DistributedTaskBalancer(int my_rank, int total_ranks)
    : rank(my_rank), world_size(total_ranks) 
{
    local_tasks.reserve(2000);
    if (ENABLE_BALANCING) {
        responder_thread = std::thread(&DistributedTaskBalancer::response_handler, this);
    }
}

DistributedTaskBalancer::~DistributedTaskBalancer() {
    if (ENABLE_BALANCING) {
        running = false;
        int stop = -1;
        MPI_Send(&stop, 1, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
        if (responder_thread.joinable()) responder_thread.join();
    }
}

void DistributedTaskBalancer::execute(int num_iterations) {
    double start_time = MPI_Wtime();
    double total_imbalance = 0.0;

    long long base_tasks = TOTAL_TASKS / world_size;
    if (rank == world_size - 1) {
        base_tasks += TOTAL_TASKS % world_size;
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        double my_weight = 0.0; //суммарный вес решенных задач для LIF
        //вес одной задачи на этой итерации
        int task_weight = calculate_task_weight(iter);

        {
            std::lock_guard<std::mutex> lock(mtx);
            pending = base_tasks;
            local_tasks.resize(base_tasks);
            std::fill(local_tasks.begin(), local_tasks.end(), task_weight);
        }

        while (true) {
            //решение задач порциями по 1200
            std::vector<int> chunk;

            {
                std::lock_guard<std::mutex> lock(mtx);
                if (pending > 0) {
                    long long take = std::min(pending, 1200LL);
                    chunk.reserve(take);
                    for (long long i = 0; i < take; ++i) {
                        chunk.push_back(local_tasks[pending - 1]);
                        pending--;
                    }
                }
            }

            if (!chunk.empty()) {
                for (int w : chunk) {
                    compute_task(w);
                    my_weight += w;
                    completed++;
                }
            } else if (ENABLE_BALANCING && !try_steal_work()) {
                break;
            } else if (!ENABLE_BALANCING) {
                break;  // без балансировки просто выход, когда задачи кончились
            }
        }

        //LIF = максимальный вес / средний вес
        double max_w = 0.0, sum_w = 0.0;
        MPI_Allreduce(&my_weight, &max_w, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&my_weight, &sum_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double imbalance = (sum_w > 0.0) ? max_w / (sum_w / world_size) : 1.0;
        total_imbalance += imbalance;
    }

    double elapsed = MPI_Wtime() - start_time;
    if (rank == 0) {
        std::cout << "Average Load Imbalance: " << (total_imbalance / num_iterations) << "\n";
    }
    std::cout << "Rank " << rank << ": " << completed 
              << " tasks | Time: " << elapsed << " s\n";
}

//вычисляется вес задачи для данного процесса на данной итерации
int DistributedTaskBalancer::calculate_task_weight(int iter) {
    int center = iter % world_size; //вершина пирамиды самая загруженная
    long long total_coeffs = 0;

    for (int i = 0; i < world_size; ++i) {
        total_coeffs += (world_size - std::abs(i - center));
    }

    long long my_coeff = world_size - std::abs(rank - center);
    long long my_total_weight = (TOTAL_WEIGHT * my_coeff) / total_coeffs;
    long long base_tasks = TOTAL_TASKS / world_size;
    if (rank == world_size - 1) {
        base_tasks += TOTAL_TASKS % world_size;
    }

    return base_tasks > 0 ? static_cast<int>(my_total_weight / base_tasks) : 1000;
}

//эмуляция вычислительной работы 
void DistributedTaskBalancer::compute_task(int weight) {
    double result = 0.0;
    for (int i = 0; i < weight; ++i) {
        result += std::sin(static_cast<double>(i)) * std::cos(static_cast<double>(i)) 
                + std::sqrt(static_cast<double>(i));
    }
    volatile double sink = result;
}

//ожидание запроса на кражу задач
void DistributedTaskBalancer::response_handler() {
    while (running) {
        int requester;
        MPI_Status status;
        MPI_Recv(&requester, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);

        if (requester == -1) break;

        std::lock_guard<std::mutex> lock(mtx);

        //отдаем половину от того что можем отдать 
        if (pending > MIN_SHARE) {
            long long share = pending / 2;
            pending -= share;

            MPI_Send(&share, 1, MPI_INT, requester, RESPONSE_TAG, MPI_COMM_WORLD);
            MPI_Send(local_tasks.data() + pending, share, MPI_INT, requester, DATA_TAG, MPI_COMM_WORLD);
        } else {
            int zero = 0;
            MPI_Send(&zero, 1, MPI_INT, requester, RESPONSE_TAG, MPI_COMM_WORLD);
        }
    }
}

bool DistributedTaskBalancer::try_steal_work() {
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned>(time(nullptr) + rank * 17));
        seeded = true;
    }

    //8 попыток украсть задачу чтобы минимизировать накладные расходы
    int start = rand() % world_size;
    int attempts = std::min(world_size - 1, 8);

    for (int i = 0; i < attempts; ++i) {
        int victim = (start + i) % world_size;
        if (victim == rank) continue;

        int received = 0;
        MPI_Send(&rank, 1, MPI_INT, victim, REQUEST_TAG, MPI_COMM_WORLD);
        MPI_Recv(&received, 1, MPI_INT, victim, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (received > 0) {
            std::lock_guard<std::mutex> lock(mtx);
            local_tasks.resize(received);
            MPI_Recv(local_tasks.data(), received, MPI_INT, victim, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pending = received;
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <iterations>\n";
        return 1;
    }

    int iterations = std::stoi(argv[1]);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Warning: MPI_THREAD_MULTIPLE not supported, using level " << provided << "\n";
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        DistributedTaskBalancer balancer(rank, size);
        balancer.execute(iterations);
    }

    MPI_Finalize();
    return 0;
}
