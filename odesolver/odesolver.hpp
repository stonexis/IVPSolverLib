#pragma once
#include <cstddef>
#include <utility>
#include <functional>
#include "span.hpp"
#include "alignedbuffer.hpp"

namespace ode{
    
    template<class Span>
    struct ReturnBuffer {
        using Scalar = typename Span::Scalar;

        //Для владения обьектом вызывающей стороной
        AlignedBuffer<Scalar> x_buf, y1_buf, y2_buf;
        bool success{false};

        //Для легкого доступа к данным
        Span  x () noexcept { return { x_buf.data(),  x_buf.size()  }; }
        Span y1 () noexcept { return { y1_buf.data(), y1_buf.size() }; }
        Span y2 () noexcept { return { y2_buf.data(), y2_buf.size() }; }
    };


    /**
    * @brief Функция для решения ОДУ 2-го порядка
    * @tparam ExpDim Ожидаемый (инициализирующий) размер сетки
    * @tparam Span Предпочитаемвй тип оболочки
    * @tparam Rhs Тип правой части ОДУ (Выводится компилятором автоматически)
    * @tparam Kernel Тип функции, производящей шаг интегрирования (Выводится компилятором автоматически)
    * @param F Правая часть ДУ
    * @param a Левый конец интегрирования
    * @param b Правый конец интегрирования
    * @param q0 Начальное условние для y
    * @param q1 Начальное условние для y'
    * @param eps_tol Требуемая точность решения (Не менее 1e-6)
    * @param kernel Функция производащая шаг интегрирования
    * @return Пара сеток y и y'
    */
    template<std::size_t ExpDim, typename Span, class Rhs, class Kernel>
    ReturnBuffer<Span> integrate_adaptive(
                                        const Rhs&  F,                  
                                        typename Span::Scalar a,  
                                        typename Span::Scalar b,
                                        typename Span::Scalar q0,
                                        typename Span::Scalar q1,
                                        typename Span::Scalar eps_tol,     
                                        Kernel&& kernel                 
                                    );

    //-----------  Ядра ------------------

    // нет накладных расходов, тк order - константа времени компиляции,
    // а значит стурктура эквивалентна свободной функции
    template<typename Span, class Rhs>
    struct EulerStepper {
        static constexpr std::size_t order = 1;   
        
        void operator()(
                    Span x, Span y1, Span y2,
                    const Rhs& F,
                    typename Span::Scalar q0,
                    typename Span::Scalar q1
                ) const;
    };

    template<typename Span, class Rhs>
    struct EulerRecountStepper {
        static constexpr std::size_t order = 1;   
        
        void operator()(
                    Span x, Span y1, Span y2,
                    const Rhs& F,
                    typename Span::Scalar q0,
                    typename Span::Scalar q1
                ) const;
    };

    template<typename Span, class Rhs>
    struct RK2Stepper {
        static constexpr std::size_t order = 2;   
        
        void operator()(
                    Span x, Span y1, Span y2,
                    const Rhs& F,
                    typename Span::Scalar q0,
                    typename Span::Scalar q1
                ) const;
    };

} //namespace ode


namespace utils{
    template <typename Span>
    inline void fill_uniform_grid(Span& grid, typename Span::Scalar a, typename Span::Scalar b);

    template<class Span>
    inline void grinding_grid_without_recount(Span grid_old, std::size_t ratio, typename Span::Scalar a, typename Span::Scalar b);

    /**
    * @brief Функция для проверки критерия Ричардсона ||Y_h - Y_2h||/(q^p - 1) < eps
    * @tparam Span Тип обертка для пары (указатель, размер массива)
    * @param solution_h Решение на более мелкой сетке (пара y, y')
    * @param solution_h Решение на более крупной сетке (пара y, y')
    * @param tol Допустимая точность
    * @param order Порядок метода
    * @param q Отношение размера(количества узлов) большей сетки к меньшей, по умолчанию = 2
    */
    template<class Span>
    bool check_richardson_criterion(
                                Span Yh,
                                Span Yh_der,
                                Span Y2h,
                                Span Y2h_der,
                                typename Span::Scalar tol,       
                                std::size_t order,
                                std::size_t q=2        
                            );









}  //namespace utils


#include "odedetail.tpp"