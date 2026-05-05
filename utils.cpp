#include "utils.hpp"

bool is_prime(std::uint64_t x) {
    if (x < 2) return false;
    if (x % 2 == 0) return x == 2;
    for (std::uint64_t d = 3; d * d <= x; d += 2) {
        if ((x % d) == 0) return false;
    }
    return true;
}

std::uint64_t find_next_prime(std::uint64_t x) {
    if (x % 2 == 0) x++;
    while (!is_prime(x)) x += 2;
    return x;
}

//возведение в степень по модулю через двоичное представление показателя
std::uint64_t pow_mod(std::uint64_t base, std::uint64_t exp, std::uint64_t mod) {
    if (mod == 1) return 0;
    std::uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 != 0) {
            const unsigned __int128 mul = (unsigned __int128)res * (unsigned __int128)base;
            res = static_cast<std::uint64_t>(mul % mod);
        }
        const unsigned __int128 sq = (unsigned __int128)base * (unsigned __int128)base;
        base = static_cast<std::uint64_t>(sq % mod);
        exp >>= 1;
    }
    return res;
}

std::uint64_t solve_dlog(std::uint64_t p, std::uint64_t g, std::uint64_t y) {
    std::uint64_t cur = 1;
    for (std::uint64_t x = 0; x < p; ++x) {
        if (cur == y) return x;
        const unsigned __int128 mul = (unsigned __int128)cur * (unsigned __int128)g;
        cur = static_cast<std::uint64_t>(mul % p);
    }
    return std::numeric_limits<std::uint64_t>::max();
}

//минимальная граница с заданным количеством бит
static std::uint64_t bit_min(int bits) {
    if (bits <= 1) {
        return 2;
    }
    return (1 << (bits - 1));
}

//максимальная граница с заданным количеством бит 
static std::uint64_t bit_max(int bits) {
    if (bits >= 63) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (1 << bits) - 1;
}

static std::uint64_t random_prime(int bits, std::mt19937_64& rng) {
    const std::uint64_t lo = bit_min(bits);  
    const std::uint64_t hi = bit_max(bits);  
    std::uniform_int_distribution<std::uint64_t> dist(lo, hi);
    std::uint64_t seed = dist(rng);    
    std::uint64_t p = find_next_prime(seed);       
    if (p > hi) {                            
        p = find_next_prime(lo + 1);              
    }
    return p;
}

std::vector<Task> make_wave(int total_tasks, std::mt19937_64& rng) {
    std::vector<Task> out;                             
    out.reserve(static_cast<std::size_t>(total_tasks));
    constexpr std::uint64_t g = 5;                  
    for (uint64_t i = 0; i < total_tasks; i++) {
        //генерация случайного простого p (29 бит)
        std::uint64_t p = random_prime(29, rng);
        //выборка случайного секретного x (от 1 до p-2)
        std::uniform_int_distribution<std::uint64_t> xdist(1, p - 2);
        const std::uint64_t x = xdist(rng);
        //вычисление y = g^x mod p
        const std::uint64_t y = pow_mod(g, x, p);
        out.push_back(Task{i, p, y});
    }
    return out;
}
