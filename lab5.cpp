#include "utils.hpp"

#include <mpi.h>
#include <pthread.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

//константы для MPI тегов
constexpr int TAG_WAVE_START = 1;
constexpr int TAG_WAVE_DATA = 2;
constexpr int TAG_WAVE_STOP = 3;
constexpr int TAG_RESULT = 4;
constexpr int TAG_STEAL_REQ = 5;
constexpr int TAG_STEAL_COUNT = 6;
constexpr int TAG_STEAL_DATA = 7;
constexpr int TAG_SHUTDOWN = 8;

//класс управления очередью задач и синхронизации
class SolverState {
 public:
  SolverState(int world_rank, int world_size)
      : rank_(world_rank), world_size_(world_size) {}

  //количество задач в очереди
  int queue_size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(tasks_.size());
  }

  //извлечение задачи из начала очереди
  bool pop_task(Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
      return false;
    }
    task = tasks_.front();
    tasks_.pop_front();
    return true;
  }

  //очистка очереди
  void clear_tasks() {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.clear();
  }

  //добавление задач в конец очереди
  void push_tasks(const std::vector<Task>& tasks) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& t : tasks) {
      tasks_.push_back(t);
    }
  }

  //кража задач с конца очереди
  std::vector<Task> request_tasks(size_t requested) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int give = std::min(requested, tasks_.size());
    std::vector<Task> out;
    out.reserve(give);
    for (int i = 0; i < give; i++) {
      out.push_back(tasks_.back());
      tasks_.pop_back();
    }
    return out;
  }

  //проверка активности волны
  bool is_wave_active() const { return is_wave_active_.load(); }
  //проверка завершения работы
  bool is_shutting_down() const { return is_shutdown_.load(); }

  //установка флага активности волны
  void set_wave_active(bool value) {
    is_wave_active_.store(value);
    wave_cv_.notify_all();
  }
  
  //установка флага завершения
  void set_shutting_down(bool value) {
    is_shutdown_.store(value);
    wave_cv_.notify_all();
  }

  //ожидание начала волны или завершения
  void wait_until_active_or_shutdown() {
    std::unique_lock<std::mutex> lock(cv_mutex_);
    wave_cv_.wait(
        lock, [this] { return is_shutdown_.load() || is_wave_active_.load(); });
  }

  int world_rank() const { return rank_; }
  int world_size() const { return world_size_; }

 private:
  int rank_ = -1;
  int world_size_ = 0;
  std::deque<Task> tasks_; //очередь задач
  std::mutex mutex_; //мьютекс для защиты очереди
  std::condition_variable wave_cv_; //условная переменная для ожидания волны
  std::mutex cv_mutex_; //мьютекс для условной переменной
  std::atomic<bool> is_wave_active_{false}; //флаг активности волны
  std::atomic<bool> is_shutdown_{false}; //флаг завершения
};

//отправка массива задач
void send_tasks(const Task* tasks, int count, int dst, int tag) {
  MPI_Send(tasks, sizeof(Task) * count, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
}

//прием задач
std::vector<Task> receive_tasks(int count, int src, int tag) {
  std::vector<Task> tasks(count);
  MPI_Recv(tasks.data(), sizeof(Task) * count, MPI_BYTE, src, tag,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return tasks;
}

//логирование
void append_log(int rank, const std::string& text) {
  std::ofstream outfile("log_" + std::to_string(rank) + ".txt", std::ios::app);
  outfile << "[rank " << rank << "] " << text << "\n";
  outfile.close();
}

//кража задач у соседей
bool steal_tasks(SolverState& state) {
  for (int peer = 1; peer < state.world_size(); peer++) {
    if (peer == state.world_rank()) {
      continue;
    }

    const int my_count = state.queue_size();
    int request_count = my_count;
    MPI_Send(&request_count, 1, MPI_INT, peer, TAG_STEAL_REQ, MPI_COMM_WORLD);

    int got = 0;
    MPI_Status steal_status{};
    MPI_Recv(&got, 1, MPI_INT, MPI_ANY_SOURCE, TAG_STEAL_COUNT, MPI_COMM_WORLD,
             &steal_status);

    int sender = steal_status.MPI_SOURCE;
    if (steal_status.MPI_SOURCE == state.world_rank() || got < 0) {
      return false;
    }
    if (got == 0) {
      continue;
    }
    std::vector<Task> tasks(static_cast<std::size_t>(got));
    MPI_Status steal_status2{};
    MPI_Recv(tasks.data(), static_cast<int>(sizeof(Task) * got), MPI_BYTE,
             MPI_ANY_SOURCE, TAG_STEAL_DATA, MPI_COMM_WORLD, &steal_status2);
    if (steal_status2.MPI_SOURCE == state.world_rank()) {
      return false;
    }

    state.push_tasks(tasks);
    append_log(state.world_rank(), "stole " + std::to_string(got) +
                                       " tasks from rank " +
                                       std::to_string(sender));
    return true;
  }

  return false;
}

//пробуждение потока server
void wake_stealer(int world_rank) {
  int wake_count = -1;
  MPI_Request req;
  MPI_Isend(&wake_count, 1, MPI_INT, world_rank, TAG_STEAL_COUNT,
            MPI_COMM_WORLD, &req);
}

//поток, решающий задачи
void* worker_thread(void* arg) {
  SolverState& state = *static_cast<SolverState*>(arg);

  while (!state.is_shutting_down()) {
    if (!state.is_wave_active()) {
      state.wait_until_active_or_shutdown();
      continue;
    }

    Task task{};
    if (state.pop_task(task)) {
      constexpr std::uint64_t g = 5ULL;

      const auto start = std::chrono::high_resolution_clock::now();
      const std::uint64_t x = solve_dlog(task.p, g, task.y);
      const auto end = std::chrono::high_resolution_clock::now();

      std::uint64_t payload[4] = {task.id, task.p, task.y, x};
      MPI_Send(payload, 4, MPI_UINT64_T, 0, TAG_RESULT, MPI_COMM_WORLD);

      const int duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      append_log(state.world_rank(),
                 "solved task id=" + std::to_string(task.id) + " in " +
                     std::to_string(duration_ms) + " ms");
      continue;
    }
    steal_tasks(state);
  }
  return nullptr;
}

//поток, отдающий задачи
void* server_thread(void* arg) {
  SolverState& state = *static_cast<SolverState*>(arg);
  while (true) {
    int requester_count = 0;
    MPI_Status status{};
    MPI_Recv(&requester_count, 1, MPI_INT, MPI_ANY_SOURCE, TAG_STEAL_REQ,
             MPI_COMM_WORLD, &status);

    const bool self_wake = status.MPI_SOURCE == state.world_rank();
    if (self_wake) {
      break;
    }
    if (state.is_shutting_down()) {
      int count = 0;
      MPI_Send(&count, 1, MPI_INT, status.MPI_SOURCE, TAG_STEAL_COUNT,
               MPI_COMM_WORLD);
      continue;
    }
    const int local_count = state.queue_size();
    int to_share = 0;
    if (local_count > requester_count + 1) {
      to_share = (local_count - requester_count) / 2;
    }

    auto stolen = state.request_tasks(to_share);
    int count = static_cast<int>(stolen.size());
    MPI_Send(&count, 1, MPI_INT, status.MPI_SOURCE, TAG_STEAL_COUNT,
             MPI_COMM_WORLD);

    if (count > 0) {
      send_tasks(stolen.data(), count, status.MPI_SOURCE, TAG_STEAL_DATA);
      append_log(state.world_rank(),
                 "shared " + std::to_string(count) + " of " +
                     std::to_string(local_count) +
                     " tasks to rank=" + std::to_string(status.MPI_SOURCE));
    }
  }
  return nullptr;
}

//основная функция рабочего процесса
void run_solver(int world_rank, int world_size) {
  SolverState state(world_rank, world_size);

  pthread_t worker{};
  pthread_t server{};
  pthread_create(&worker, nullptr, worker_thread, &state);
  pthread_create(&server, nullptr, server_thread, &state);

  while (true) {
    MPI_Status status{};
    int meta[2] = {0, 0};
    MPI_Recv(meta, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == TAG_WAVE_START) {
      const int wave_id = meta[0];
      const int count = meta[1];
      state.clear_tasks();

      if (count > 0) {
        state.push_tasks(receive_tasks(count, 0, TAG_WAVE_DATA));
      }

      state.set_wave_active(true);
      append_log(world_rank, "wave " + std::to_string(wave_id) +
                                 " start, local tasks=" +
                                 std::to_string(state.queue_size()));
      continue;
    }

    if (status.MPI_TAG == TAG_WAVE_STOP) {
      state.set_wave_active(false);
      state.clear_tasks();
      wake_stealer(world_rank);
      append_log(world_rank, "wave stop");
      continue;
    }

    if (status.MPI_TAG == TAG_SHUTDOWN) {
      append_log(world_rank, "shutdown");
      break;
    }
  }
  state.set_shutting_down(true);
  state.set_wave_active(false);

  append_log(world_rank, "wake 1");
  wake_stealer(world_rank);

  append_log(world_rank, "wake 2");
  int wake_request = -1;
  MPI_Request req;
  MPI_Isend(&wake_request, 1, MPI_INT, world_rank, TAG_STEAL_REQ,
           MPI_COMM_WORLD, &req);

  append_log(world_rank, "join threads");
  pthread_join(worker, nullptr);
  pthread_join(server, nullptr);
  append_log(world_rank, "stop");
}

//функция генератора (мастер процесс)
void run_generator(int world_size, int waves, int tasks_per_worker) {
  const int num_workers = world_size - 1;
  if (num_workers <= 0) {
    std::cerr << "Need at least 2 MPI processes\n";
    return;
  }
  std::mt19937_64 rng(3331);

  for (int wave = 0; wave < waves; wave++) {
    const int total_tasks = tasks_per_worker * num_workers;
    auto tasks = make_wave(total_tasks, rng);

    int remaining = total_tasks;
    int offset = 0;
    for (int worker_index = 0; worker_index < num_workers; worker_index++) {
      const int rank = worker_index + 1;
      const int ranks_left = num_workers - worker_index;
      int count = 0;
      if (ranks_left == 1) {
        count = remaining;
      } else {
        std::uniform_int_distribution<int> pick_count(0, remaining);
        count = pick_count(rng);
      }

      remaining -= count;
      int start_meta[2] = {wave, count};
      MPI_Send(start_meta, 2, MPI_INT, rank, TAG_WAVE_START, MPI_COMM_WORLD);

      if (count > 0) {
        send_tasks(tasks.data() + offset, count, rank, TAG_WAVE_DATA);
      }
      append_log(0, "wave " + std::to_string(wave) + ": sent " +
                        std::to_string(count) +
                        " tasks to rank=" + std::to_string(rank));
      offset += count;
    }
    //сборка результатов
    int received = 0;
    while (received < total_tasks) {
      MPI_Status status{};
      std::uint64_t result[4] = {0ULL, 0ULL, 0ULL, 0ULL};
      MPI_Recv(result, 4, MPI_UINT64_T, MPI_ANY_SOURCE, TAG_RESULT,
               MPI_COMM_WORLD, &status);

      append_log(0, "result from rank " + std::to_string(status.MPI_SOURCE) +
                        " task=" + std::to_string(result[0]) + " remaining " +
                        std::to_string(total_tasks - received));
      received += 1;
    }
    if (wave != waves - 1) {
      for (int rank = 1; rank < world_size; rank++) {
        int meta[2] = {wave, 0};
        MPI_Send(meta, 2, MPI_INT, rank, TAG_WAVE_STOP, MPI_COMM_WORLD);
      }
    }
  }
  //отправка shutdown всем рабочим
  for (int rank = 1; rank < world_size; rank++) {
    int meta[2] = {-1, 0};
    MPI_Send(meta, 2, MPI_INT, rank, TAG_SHUTDOWN, MPI_COMM_WORLD);
  }
  append_log(0, "done");
}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
    std::cerr << "threads not supported\n";
    MPI_Finalize();
    return 1;
  }
  int world_rank = -1;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int waves = std::max(1, std::atoi(argv[1]));
  int tasks_per_worker = std::max(4, std::atoi(argv[2]));
  if (world_rank == 0) {
    run_generator(world_size, waves, tasks_per_worker);
  } else {
    run_solver(world_rank, world_size);
  }
  MPI_Finalize();
  return 0;
}
