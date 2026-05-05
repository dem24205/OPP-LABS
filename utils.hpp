#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <random>
#include <vector>

struct Task {
    std::uint64_t id; //номер задания
    std::uint64_t p; //простое число - модуль 
    std::uint64_t y; //результат возведения в степень
};

//поиск ближайшего простого числа
bool is_prime(std::uint64_t x);
std::uint64_t find_next_prime(std::uint64_t x);

//возведение в степень по модулю
std::uint64_t pow_mod(std::uint64_t base, std::uint64_t exp, std::uint64_t mod);

//решение дискретного логарифма (g^x mod p = y)
std::uint64_t solve_dlog(std::uint64_t p, std::uint64_t g, std::uint64_t y);

//генерация волны заданий
std::vector<Task> make_wave(int total_tasks, std::mt19937_64& rng);

#endif
