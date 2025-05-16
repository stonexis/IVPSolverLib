#pragma once
#include <cstddef>
#include <memory>

//(предполагается ТОЛЬКО совместное использование с памятью выделенной AlignmentBuffer)
template<typename T>
class ali_span {
    T*           p_;
    std::size_t  n_;
public:
    using Scalar = T;

    constexpr ali_span() noexcept : p_(nullptr), n_(0) {}
    constexpr ali_span(T* p, std::size_t n) noexcept : p_(p), n_(n) {}

    constexpr std::size_t size() const noexcept { return n_; }
    //Доступ  к данным только через предположение о выравнивании
    T* data()             noexcept { return std::assume_aligned<64, T>(p_);}
    const T* data() const noexcept { return std::assume_aligned<64, T>(p_);}

    T& operator[](std::size_t i)       noexcept { return p_[i]; }
    T  operator[](std::size_t i) const noexcept { return p_[i]; }
};