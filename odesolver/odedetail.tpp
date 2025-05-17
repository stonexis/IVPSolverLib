#pragma once
#include <string_view>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <iostream>
#include "span.hpp"
#include "alignedbuffer.hpp"


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
            if (utils::check_richardson_criterion(yh2, ydh2, yh, ydh, eps_tol, kernel.order)){ 
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
    template<std::size_t GridDim, class Span, class Rhs, class Kernel>
    ReturnBuffer<Span> integrate_freeze(
                                    const Rhs&  F,
                                    typename Span::Scalar a,
                                    typename Span::Scalar b,
                                    typename Span::Scalar q0,
                                    typename Span::Scalar q1,
                                    Kernel&& kernel
                                ){
    using Scalar = typename Span::Scalar;

    AlignedBuffer<Scalar> xh_buf(GridDim);
    AlignedBuffer<Scalar> yh_buf(GridDim);
    AlignedBuffer<Scalar> ydh_buf(GridDim);

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





    

    

}   //namespace ode

namespace utils {
    
    template<class Span>
    bool check_richardson_criterion(
                            Span Yh,
                            Span Yh_der,
                            Span Y2h,
                            Span Y2h_der,
                            typename Span::Scalar tol,       
                            std::size_t           order,
                            std::size_t           q        
                        ){
        using S = typename Span::Scalar;
        const S powq = std::pow(S(q), S(order)); // q^p предвычисление
        const std::size_t small_size = Y2h.size();
        const std::size_t big_size = Yh.size();
        
        //Создаем беспсевдонимный тип (disable alias type) для потенциального ускорения
        const S* __restrict big   = Yh.data();
        const S* __restrict big_d = Yh_der.data();
        const S* __restrict sml   = Y2h.data();
        const S* __restrict sml_d = Y2h_der.data();

        //||Yh - Y2h|| без доп выделений памяти
        S diff_func = S(0), diff_der = S(0); //Инициализация суммы
        //Параллелизм на уровне данных (Векторизация), так же сообщаем что данные уже выровнены, и инициализируем сумму нулем, каждый векторный сегмент сложит туда свой результат
        #pragma omp simd aligned(big,big_d,sml,sml_d:64) reduction(+:diff_func,diff_der)
        for (std::size_t i = 0; i < small_size; ++i) {
            std::size_t j = i*q;
            diff_func += std::abs(big[i*q]   - sml  [i]); //Сумма модулей разностей компонент сеток функций (из большей сетки берутся значения с пропуском q)
            diff_der  += std::abs(big_d[i*q] - sml_d[i]); //Сумма модулей разностей компонент сеток производных (из большей сетки берутся значения с пропуском q)
            
        }
        
        //||Y_h - Y_2h||/(q^p - 1)
        S err  = diff_func   / ( S(big_size-1) * (powq - S(1)));
        S errd = diff_der / ( S(big_size-1) * (powq - S(1))); 
       
        return err <= tol; //Можно заменить на максимум из err errd для отслеживания точности производной
    }

    





    template<class Grid>
    using pack_norms = std::array<typename Grid::Scalar, 6>;

    enum Norms : std::size_t { // Не вызывает никаких накладных расходов, поскольку на этапе компиляции преобразуется в числа
        L_1_abs,   //0
        L_2_abs,   //1  
        L_inf_abs, //2
        
        L_1_rel,   //3
        L_2_rel,   //4
        L_inf_rel  //5
    };

    template<class BiggerGrid, class SmallerGrid>
    pack_norms<BiggerGrid> deviations(const BiggerGrid& bigger, const SmallerGrid& smaller){
        static_assert((bigger.size - 1) % (smaller.size - 1) == 0, "Grids must be didvided into one another");
        pack_norms<BiggerGrid> norms;
        constexpr std::size_t multiplier = (bigger.size - 1) / (smaller.size - 1);
        SmallerGrid sparse_bigger;
        
        //Прорядить сетку для сравнения
        for(std::size_t i = 0; i < smaller.size; i++)
            sparse_bigger[i] = bigger[i * multiplier];
        

        norms[0] = sparse_bigger.calc_1_norm_difference(smaller);
        norms[1] = sparse_bigger.calc_2_norm_difference(smaller);
        norms[2] = sparse_bigger.calc_inf_norm_difference(smaller);

        norms[3] = norms[0] / sparse_bigger.l1_norm();
        norms[4] = norms[1] / sparse_bigger.l2_norm();
        norms[5] = norms[2] / sparse_bigger.linf_norm(); 

        return norms;
    }


    template <class Span>
    void write_to_file_grid(std::ofstream &out, Span arr, std::string_view name_arr){
        out << "\"" << name_arr << "\"" << ": ["; //форматируем файл под json
        for (std::size_t i = 0; i < arr.size(); ++i) {
            out << arr[i];
            if (i != arr.size() - 1) 
                out << ", ";
        }
        out << "]";
    }

    template <class Pair, class... Rest>
    void write_data_to_json_impl(std::ofstream &out, const Pair& GridAndName, Rest... rest){
        write_to_file_grid(out, GridAndName.first, GridAndName.second);
    
        if constexpr (sizeof...(rest) > 0) { //Остановка рекурсии
            out << ",\n";
            write_data_to_json_impl(out, rest...);
        }

    }

    template <typename Span>
    inline void fill_uniform_grid(Span& grid, typename Span::Scalar a, typename Span::Scalar b){
        const std::size_t size = grid.size();
        typename Span::Scalar step = std::abs(b - a) / (size - 1);
        for (std::size_t i = 0; i < size; i++) 
            grid[i] = a + step * i; // Заполняем значения, включая последний узел, равный b
        if (grid[size - 1] != b)
            grid[size - 1] = b;
    }

    template<class Span>
    inline void grinding_grid_without_recount(Span grid_old, std::size_t ratio, typename Span::Scalar a, typename Span::Scalar b){
        //Предвычисления
        const std::size_t size_old = grid_old.size();
        const std::size_t size_new = ratio * grid_old.size();
        const typename Span::Scalar step_new = abs(b - a) / (size_new - 1);

        //Новая сетка
        typename Span::Scalar* grinded_grid = new typename Span::Scalar[size_new];

        for (std::size_t i = 0; i < size_old - 1; i++) {
            grinded_grid[i * ratio] = grid_old[i]; // Старая точка
            for (std::size_t j = 1; j < ratio; j++) { // Заполнение промежуточных точек
                grinded_grid[i * ratio + j] = grid_old[i] + j * step_new; 
            }
        }
        grinded_grid[size_new - 1] = grid_old[size_old - 1]; // Последняя точка обязана совпадать

        delete [] grid_old.data();
        //Перезапись данных в том же месте
        grid_old.reset(grinded_grid, size_new);
    }






} // namespace utils

template<class... Pairs>
void write_data_to_json_file(const std::string& filename, const Pairs&... GridsAndNames){
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << "{\n";
    utils::write_data_to_json_impl(out, GridsAndNames...);
    out << "\n}\n";

    // out автоматически закроется при выходе из функции
}
























    
