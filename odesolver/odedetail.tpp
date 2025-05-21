#pragma once
#include <string_view>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <iostream>
#include <fstream>
#include "span.hpp"
#include "alignedbuffer.hpp"
#include "utils.hpp"


namespace ode{
    //---------------------- Ядра ----------------------------

    template<class Span, class Rhs>
    void EulerStepper<Span, Rhs>::operator()(
                                        Span x, Span y1, Span y2,
                                        const Rhs& F,
                                        typename Span::Scalar q0,
                                        typename Span::Scalar q1
                                    ) const{
        using S = typename Span::Scalar;
        const std::size_t size = x.size();
        const S h = std::abs(x[1] - x[0]);

        y1[0] = q0;
        y2[0] = q1;

        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        S* __restrict y1dat = y1.data();
        S* __restrict y2dat = y2.data();
        const S* __restrict xdat = x.data();

        //Пользуемся выравниванием
        #pragma omp simd aligned(xdat,y1dat,y2dat:64)
        for (std::size_t k = 1; k < size; ++k) {
            y1dat[k] = y1dat[k-1] + h * y2dat[k-1];
            y2dat[k] = y2dat[k-1] + h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);
        }

    }


    template<class Span, class Rhs>
    void EulerRecountStepper<Span, Rhs>::operator()(
                                            Span x, Span y1, Span y2,
                                            const Rhs& F,
                                            typename Span::Scalar q0,
                                            typename Span::Scalar q1
                                        ) const{
        using S = typename Span::Scalar;
        const std::size_t size = x.size();
        const S h = std::abs(x[1] - x[0]);

        y1[0] = q0;
        y2[0] = q1;

        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        S* __restrict y1dat = y1.data();
        S* __restrict y2dat = y2.data();
        const S* __restrict xdat = x.data();

        constexpr std::size_t maxiter = 2;
    
        for (std::size_t k = 1; k < size; ++k) {
            //Предиктор
            y1dat[k] = y1dat[k-1] + h * y2dat[k-1];
            y2dat[k] = y2dat[k-1] + h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);
            
            //Корректор
            #pragma omp simd aligned(xdat,y1dat,y2dat:64) //Можно использовать векторизацию
            for(std::size_t i = 0; i < maxiter; i++){
                y1dat[k] = y1dat[k-1] + h * y2dat[k];
                y2dat[k] = y2dat[k-1] + h * F(xdat[k], y1dat[k], y2dat[k]);   
            }
            
        }


    }

    template<class Span, class Rhs>
    void RK2Stepper<Span, Rhs>::operator()(
                                        Span x, Span y1, Span y2,
                                        const Rhs& F,
                                        typename Span::Scalar q0,
                                        typename Span::Scalar q1
                                    ) const{
        using S = typename Span::Scalar;
        const std::size_t size = x.size();
        const S h = std::abs(x[1] - x[0]);

        y1[0] = q0;
        y2[0] = q1;

        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        S* __restrict y1dat = y1.data();
        S* __restrict y2dat = y2.data();
        const S* __restrict xdat = x.data();

        constexpr std::size_t maxiter = 4;
    
        const S h_2 = h / 2.0; //Предвычисление
        for (std::size_t k = 1; k < size; ++k) {
            //Коэффициенты рунге-кутты
            S k1_1 = h * y2dat[k-1];
            S k1_2 = h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);

            S k2_1 = h * (y2dat[k-1] + k1_2 / 2.0);
            S k2_2 = h * F(xdat[k-1] + h_2, y1dat[k-1] + k1_1 / 2.0, y2dat[k-1] + k1_2 / 2.0);

            y1dat[k] = y1dat[k-1] + k2_1;
            y2dat[k] = y2dat[k-1] + k2_2;
            
        }

    }

    template<class Span, class Rhs>
    void RK4Stepper<Span, Rhs>::operator()(
                                        Span x, Span y1, Span y2,
                                        const Rhs& F,
                                        typename Span::Scalar q0,
                                        typename Span::Scalar q1
                                    ) const{
        using S = typename Span::Scalar;
        const std::size_t size = x.size();
        const S h = std::abs(x[1] - x[0]);

        y1[0] = q0;
        y2[0] = q1;

        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        S* __restrict y1dat = y1.data();
        S* __restrict y2dat = y2.data();
        const S* __restrict xdat = x.data();
        
        const S h_2 = h / 2.0; //Предвычисление
        for (std::size_t k = 1; k < size; ++k) {
            //Коэффициенты рунге-кутты
            S k1_1 = h * y2dat[k-1];
            S k1_2 = h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);

            S k2_1 = h * (y2dat[k-1] + k1_2 / 2.0);
            S k2_2 = h * F(xdat[k-1] + h_2, y1dat[k-1] + k1_1 / 2.0, y2dat[k-1] + k1_2 / 2.0);

            S k3_1 = h * (y2dat[k-1] + k2_2 / 2.0);
            S k3_2 = h * F(xdat[k-1] + h_2, y1dat[k-1] + k2_1 / 2.0, y2dat[k-1] + k2_2 / 2.0);

            S k4_1 = h * (y2dat[k-1] + k3_2);
            S k4_2 = h * F(xdat[k-1] + h, y1dat[k-1] + k3_1, y2dat[k-1] + k3_2);

            y1dat[k] = y1dat[k-1] + (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1) / 6.0;
            y2dat[k] = y2dat[k-1] + (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2) / 6.0;
            
        }

    }

    template<class Span, class Rhs>
    void AdamsStepper<Span, Rhs>::operator()(
                                        Span x, Span y1, Span y2,
                                        const Rhs& F,
                                        typename Span::Scalar q0,
                                        typename Span::Scalar q1
                                    ) const{
        using S = typename Span::Scalar;
        const std::size_t size = x.size();
        const S h = std::abs(x[1] - x[0]);

        y1[0] = q0;
        y2[0] = q1;

        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        S* __restrict y1dat = y1.data();
        S* __restrict y2dat = y2.data();
        const S* __restrict xdat = x.data();
        const S h_2 = h / 2.0;
        //Разгон адамса РК2
        for(std::size_t k = 1; k < 4; ++k){
            S k1_1 = h * y2dat[k-1];
            S k1_2 = h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);

            S k2_1 = h * (y2dat[k-1] + k1_2 / 2.0);
            S k2_2 = h * F(xdat[k-1] + h_2, y1dat[k-1] + k1_1 / 2.0, y2dat[k-1] + k1_2 / 2.0);

            y1dat[k] = y1dat[k-1] + k2_1;
            y2dat[k] = y2dat[k-1] + k2_2;
        }
        //Адамс
        for (std::size_t k = 3; k < size; ++k) {
            y1dat[k] = y1dat[k-1] + (h / 12.0) * (23.0 * y2dat[k-1] - 16.0 * y2dat[k-2] + 5.0 * y2dat[k-3]);
            y2dat[k] = y2dat[k-1] + (h / 12.0) * (23.0 * F(xdat[k-1], y1dat[k-1], y2dat[k-1]) - 16.0 * F(xdat[k-2], y1dat[k-2], y2dat[k-2]) + 5.0 * F(xdat[k-3], y1dat[k-3], y2dat[k-3]));
            
        }

    }

    
//-------------Подбор шага --------------------------------------------------------

    template<std::size_t ExpDim, class Span, class Rhs, class Kernel>
    ReturnBuffer<Span> integrate_adaptive(
                                    const Rhs&  F,
                                    typename Span::Scalar a,
                                    typename Span::Scalar b,
                                    typename Span::Scalar q0,
                                    typename Span::Scalar q1,
                                    typename Span::Scalar eps_tol,
                                    Kernel&& kernel
                                ){
        using Scalar = typename Span::Scalar;
        static_assert(ExpDim % 2 != 0, "ExpDim dont be divisible by 2");
        if (eps_tol < Scalar(1e-8))
            throw std::invalid_argument("eps_tol must be > 1e-10");

        constexpr std::size_t ratio = 2; //Изменение шага происходит посредством умножения на 2 количества узлов сетки (не размера шага)
        const std::size_t max_iter = 20;

        //Буфер для каждой сетки
        AlignedBuffer<Scalar> xh_buf(ExpDim);
        AlignedBuffer<Scalar> yh_buf(ExpDim);
        AlignedBuffer<Scalar> ydh_buf(ExpDim);

        std::size_t size_grid_h2 = ratio * (ExpDim - 1) + 1;

        AlignedBuffer<Scalar> xh2_buf(size_grid_h2);
        AlignedBuffer<Scalar> yh2_buf(size_grid_h2);
        AlignedBuffer<Scalar> ydh2_buf(size_grid_h2);

        //Обертка для каждой сетки для удобных манипуляций (хранение сохраняется в буфере)
        Span xh(xh_buf.data(), xh_buf.size());
        Span yh(yh_buf.data(), yh_buf.size());
        Span ydh(ydh_buf.data(), ydh_buf.size());

        Span xh2(xh2_buf.data(), xh2_buf.size());
        Span yh2(yh2_buf.data(), yh2_buf.size());
        Span ydh2(ydh2_buf.data(), ydh2_buf.size());

        //Заполняем сетки 
        utils::fill_uniform_grid(xh, a, b);
        utils::fill_uniform_grid(xh2, a, b);
        
        kernel(xh, yh, ydh, F, q0, q1);
        
        bool success = false;
        #pragma GCC unroll 4 //Запрещаем разворот цикла более чем на 4, поскольку внутри много template-swap => на O3 проблемы на этапе линковки
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            kernel(xh2, yh2, ydh2, F, q0, q1);
            if (utils::check_richardson_criterion(yh2, ydh2, yh, ydh, eps_tol, kernel.order, ratio)){ 
                success = true; 
                break; 
            }
            else if (iter == max_iter - 1){ //Если неуспех на последней итерации, делать свап и дробить дальше сетку незачем, результат проверить не получится 
                success = false;
                break;
            }
            //Критерий не пройден, измельчаем крупную сетку(посредством swap - старая мелкая становится крупной большой)
            //Сначала перекидываем обертки, затем хранилища
            std::swap(xh2, xh);  
            std::swap(yh2, yh);
            std::swap(ydh2, ydh);

            //Старая крупная сетка автоматически удалится поскольку память лежит в unique_ptr
            std::swap(xh2_buf, xh_buf);
            std::swap(yh2_buf, yh_buf);
            std::swap(ydh2_buf, ydh_buf);

            //Измельчаем мелкую сетку (тут уже обычное увеличение числа узлов)
            size_grid_h2 = ratio * (size_grid_h2 - 1) + 1;
            //Изменяем размер буферов(перевыделение памяти)
            xh2_buf.resize(size_grid_h2);
            yh2_buf.resize(size_grid_h2);
            ydh2_buf.resize(size_grid_h2);

            //Корректируем обертки
            xh2 = {xh2_buf.data(), size_grid_h2};
            yh2 = {yh2_buf.data(), size_grid_h2};
            ydh2 = {ydh2_buf.data(), size_grid_h2};

            //После подготовки заполняем сетку 
            utils::fill_uniform_grid(xh2, a, b);
        }

        ReturnBuffer<Span> res{
            .x_buf  = std::move(xh_buf),
            .y1_buf = std::move(yh_buf),
            .y2_buf = std::move(ydh_buf),
            .success= success
        };
        return res;
    }
    
    template<class Span, class Rhs, class Kernel>
    ReturnBuffer<Span> integrate_freeze(
                                    const Rhs&  F,
                                    typename Span::Scalar a,
                                    typename Span::Scalar b,
                                    typename Span::Scalar q0,
                                    typename Span::Scalar q1,
                                    Kernel&& kernel,
                                    std::size_t grid_size
                                ){
        using Scalar = typename Span::Scalar;

        AlignedBuffer<Scalar> xh_buf(grid_size);
        AlignedBuffer<Scalar> yh_buf(grid_size);
        AlignedBuffer<Scalar> ydh_buf(grid_size);

        Span xh(xh_buf.data(), xh_buf.size());
        Span yh(yh_buf.data(), yh_buf.size());
        Span ydh(ydh_buf.data(), ydh_buf.size());

        utils::fill_uniform_grid(xh, a, b);
        kernel(xh, yh, ydh, F, q0, q1);

        ReturnBuffer<Span> res{
            .x_buf  = std::move(xh_buf),
            .y1_buf = std::move(yh_buf),
            .y2_buf = std::move(ydh_buf),
            .success= true
        };
        return res;
    
    }

    template<class Span, class Rhs, class Kernel>
    void compare_results(
                        const Rhs&  F,
                        typename Span::Scalar a,
                        typename Span::Scalar b,
                        typename Span::Scalar q0,
                        typename Span::Scalar q1,
                        Kernel&& kernel,
                        std::size_t opt_N_h,
                        const std::string_view& method_name,
                        const std::string& filename,
                        bool header
                    ){
        std::ofstream out;
        if (header == true){//Требуется хедер => открываем для перезаписи
            out.open(filename);
            if (!out.is_open())
                throw std::runtime_error("Cannot open " + filename);
        }
        else{ //Не требуется => для дозаписи
            out.open(filename, std::ios::app);
            if (!out.is_open())
                throw std::runtime_error("Cannot open " + filename);
        }
        using S = typename Span::Scalar;
        //Количество узлов для сетки с шагом 2h от оптимального
        const std::size_t opt_N_2h = (opt_N_h - 1) / 2 + 1;
        //Количество узлов для сетки с шагом h/2 от оптимального
        const std::size_t opt_N_h2 = 2 * (opt_N_h - 1) + 1;
                       
        auto result_2h = ode::integrate_freeze<Span>(F, a, b, q0, q1, kernel, opt_N_2h);
        auto result_h = ode::integrate_freeze<Span>(F, a, b, q0, q1, kernel, opt_N_h);
        auto result_h2 = ode::integrate_freeze<Span>(F, a, b, q0, q1, kernel, opt_N_h2);

        //Попарные сравнения
        auto dev_2h_y1 = utils::deviations(result_2h.y1(), result_h.y1());
        auto dev_2h_y2 = utils::deviations(result_2h.y2(), result_h.y2());
        
        auto dev_h2_y1 = utils::deviations(result_h.y1(), result_h2.y1());
        auto dev_h2_y2 = utils::deviations(result_h.y2(), result_h2.y2());

        //Оптимальный шаг для записи в файл
        S h = result_h.x()[1] - result_h.x()[0];
        utils::write_norms_table<Span>(out, h, std::make_tuple(method_name, "y1_2h", dev_2h_y1), header);
        utils::write_norms_table<Span>(out, h, std::make_tuple(method_name, "y2_2h", dev_2h_y2), false);
        utils::write_norms_table<Span>(out, h, std::make_tuple(method_name, "y1_h2", dev_h2_y1), false);
        utils::write_norms_table<Span>(out, h, std::make_tuple(method_name, "y2_h2", dev_h2_y2), false);
    }


}   //namespace ode


























    
