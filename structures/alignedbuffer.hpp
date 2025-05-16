#pragma once
#include <cstdlib>
#include <memory>

template<typename T>
class AlignedBuffer {
    struct FreeAligned { //aligned_alloc удаляется только при помощи free
        void operator()(void* p) const noexcept { std::free(p); }
    };
    std::unique_ptr<T, FreeAligned> ptr_;
    std::size_t                     n_ = 0;

    static T* alloc(std::size_t n) {
        if (n == 0) 
            return nullptr;
        void* p = std::aligned_alloc(64, n * sizeof(T));
        if (!p) //На случай если нельзя выделить выровненную память (size не делится на alignment)
            throw std::bad_alloc{};
        return static_cast<T*>(p);
    }
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n) : ptr_(alloc(n)), n_(n) {}
    AlignedBuffer(AlignedBuffer&&)            noexcept = default;
    AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;

    void resize(std::size_t n) {                 
        if (n > n_){ //если меньше то все остается как есть
            ptr_.reset(alloc(n)); //отбираем захвачиенный unique_ptr указатель и кладем вместо него новый
            n_ = n;                 //(память выделяется так же выровненная)
        }
    }
    std::size_t size() const noexcept { return n_; }
    //Доступ  к данным только через предположение о выравнивании (поскольку они выровнены в этом классе)
    T*       data()       noexcept { return std::assume_aligned<64, T>(ptr_.get()); }
    const T* data() const noexcept { return std::assume_aligned<64, T>(ptr_.get()); }
};
