#pragma once

#include <chrono>
#include <limits>
#include <type_traits>
#include <utility>

namespace util {

// Clamp utility (C++17; std::clamp is C++17 but keeping our own for pedagogy/consistency)
template <typename T>
constexpr T clamp(T v, T lo, T hi) {
    static_assert(std::is_arithmetic<T>::value, "util::clamp requires an arithmetic type");
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Measure the minimum wall-clock time across K iterations.
// Template parameters:
// - Clock: which clock to use (defaults to steady_clock)
// - F: callable type
template <typename Clock = std::chrono::steady_clock, typename F>
long long measure_min_ms(int iters, F&& f) {
    const int k = (iters > 0) ? iters : 1;
    long long best = (std::numeric_limits<long long>::max)();
    for (int i = 0; i < k; ++i) {
        const auto t0 = Clock::now();
        std::forward<F>(f)();
        const auto t1 = Clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    if (best == (std::numeric_limits<long long>::max)()) return 0;
    return best;
}

// RAII timer: measures scope duration and writes result on destruction.
// Example:
//   long long ms = 0;
//   { util::ScopedTimer t(ms); do_work(); }
template <typename Sink, typename Clock = std::chrono::steady_clock>
class ScopedTimer {
public:
    explicit ScopedTimer(Sink& sink)
        : sink_(sink), t0_(Clock::now()) {}

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    ~ScopedTimer() noexcept {
        const auto t1 = Clock::now();
        sink_ = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0_).count();
    }

private:
    Sink& sink_;
    typename Clock::time_point t0_;
};

} // namespace util

