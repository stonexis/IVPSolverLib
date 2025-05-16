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

        constexpr std::size_t maxiter = 2;
    
        for (std::size_t k = 1; k < size; ++k) {
            //Предиктор (так же простой Эйлер)
            y1dat[k] = y1dat[k-1] + h * y2dat[k-1];
            y2dat[k] = y2dat[k-1] + h * F(xdat[k-1], y1dat[k-1], y2dat[k-1]);
            //Поскольку RK опериурет усреднение между старым и новым значением (метод трапеций), необходимо сохранять старые значения
            S y1_new = y1dat[k];
            S y2_new = y2dat[k];
            //Корректор
            const S h_2 = h/2; //Предвычисление
            #pragma omp simd aligned(xdat,y1dat,y2dat:64) //Можно использовать векторизацию
            for(std::size_t i = 0; i < maxiter; i++){
                S y1_old = y1_new;
                S y2_old = y2_new;
                y1_new = y1dat[k-1] + h_2 * (y2_old + y2_new);
                y2_new = y2dat[k-1] + h_2 * (F(xdat[k-1], y1_old, y2_old) + F(xdat[k], y1_new, y2_new));   
            }
            y1dat[k]=y1_new; 
            y2dat[k]=y2_new;
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
        static_assert(ExpDim % 2 == 0, "ExpDim must be divisible by 2");
        if (eps_tol < Scalar(1e-6))
            throw std::invalid_argument("eps_tol must be > 1e-10");

        constexpr std::size_t ratio = 2; //Изменение шага происходит посредством умножения на 2 количества узлов сетки (не размера шага)
        constexpr std::size_t max_iter = 10;

        //Буфер для каждой сетки
        AlignedBuffer<Scalar> x_buf(ExpDim);
        AlignedBuffer<Scalar> y_buf(ExpDim);
        AlignedBuffer<Scalar> yd_buf(ExpDim);

        AlignedBuffer<Scalar> x2_buf(ExpDim / 2);
        AlignedBuffer<Scalar> y2_buf(ExpDim / 2);
        AlignedBuffer<Scalar> yd2_buf(ExpDim / 2);

        auto fill_grid = [](Span g, Scalar a, Scalar b){
            const std::size_t size = g.size();
            Scalar h = std::abs(b - a) / (size - 1);
            for (std::size_t i = 0; i < size; ++i) 
                g[i] = a + i*h;
        };

        //Обертка для каждой сетки для удобных манипуляций (хранение сохраняется в буфере)
        Span x(x_buf.data(), x_buf.size());
        Span y(y_buf.data(), y_buf.size());
        Span yd(yd_buf.data(), yd_buf.size());

        Span x2(x2_buf.data(), x2_buf.size());
        Span y2(y2_buf.data(), y2_buf.size());
        Span yd2(yd2_buf.data(), yd2_buf.size());

        //Заполняем сетки 
        fill_grid(x, a, b);
        fill_grid(x2, a, b);

        //Первое решение необходимо на сетке x2 для проверки (в дальнейшем создается только на сетке x), x2 получаем от старых swap
        kernel(x2, y2, yd2, F, q0, q1);

        bool success = false;
        std::size_t grid_size = ExpDim;

        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            kernel(x, y, yd, F, q0, q1);
            if (utils::check_richardson_criterion(y, yd, y2, yd2, eps_tol, kernel.order)){ 
                success = true; 
                break; 
            }
            //Критерий не пройден, измельчаем крупную сетку(посредством swap - старая мелкая становится крупной большой)
            //Сначала перекидываем обертки, затем хранилища
            std::swap(x, x2);  
            std::swap(y, y2);
            std::swap(yd, yd2);

            //Старая крупная сетка автоматически удалится поскольку память лежит в unique_ptr
            std::swap(x_buf, x2_buf);
            std::swap(y_buf, y2_buf);
            std::swap(yd_buf, yd2_buf);

            //Измельчаем мелкую сетку (тут уже обычное увеличение числа узлов)
            grid_size *= ratio;

            //Изменяем размер буферов(перевыделение памяти)
            x_buf.resize(grid_size);
            y_buf.resize(grid_size);
            yd_buf.resize(grid_size);

            //Корректируем обертки
            x = {x_buf.data(), grid_size};
            y = {y_buf.data(), grid_size};
            yd = {yd_buf.data(), grid_size};

            //После подготовки заполняем сетку 
            fill_grid(x, a, b);
        }

        ReturnBuffer<Span> res{
            .x_buf  = std::move(x2_buf),
            .y1_buf = std::move(y2_buf),
            .y2_buf = std::move(yd2_buf),
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
        const std::size_t powq = std::pow(S(q), S(order)); // q^p предвычисление
        const std::size_t small_size = Y2h.size();
        
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
            diff_func += std::abs(big[i*q]   - sml  [i]); //Сумма модулей разностей компонент сеток функций (из большей сетки берутся значения с пропуском q)
            diff_der  += std::abs(big_d[i*q] - sml_d[i]); //Сумма модулей разностей компонент сеток производных (из большей сетки берутся значения с пропуском q)
        }

        diff_func /= ((powq - S(1)) * small_size); //||Y_h - Y_2h||/N(q^p - 1)
        diff_der  /= ((powq - S(1))* small_size);

        return std::max(diff_func, diff_der) <= tol;
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
























    
