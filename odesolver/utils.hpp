#pragma once
#include <cmath>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string_view>
#include <iomanip>
#include "span.hpp"
#include "alignedbuffer.hpp"

namespace utils {
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
    struct Norms{
        typename Span::Scalar L_1_abs;  
        typename Span::Scalar L_2_abs;     
        typename Span::Scalar L_inf_abs; 
        
        typename Span::Scalar L_1_rel;   
        typename Span::Scalar L_2_rel;   
        typename Span::Scalar L_inf_rel;
    };

    template<class Span>
    [[nodiscard]]Norms<Span> deviations(const Span& Yh, const Span& Yh2){
        const std::size_t q = (Yh2.size() - 1) / (Yh.size() - 1);
        const std::size_t gh_size = Yh.size();
        using S = typename Span::Scalar;

        const S* __restrict gh = Yh.data();
        const S* __restrict gh2 = Yh2.data(); 

        S l1_diff_abs = S(0);
        S l2_diff_abs = S(0);
        S linf_diff_abs = S(0);

        S l1_abs = S(0);
        S l2_abs = S(0);
        S linf_abs = S(0);

        #pragma omp simd aligned(gh,gh2:64) reduction(+:l1_diff_abs,l2_diff_abs,l1_abs,l2_abs)
        for(std::size_t i = 0; i < gh_size; i++){
            std::size_t j = i*q;
            //Нормы разности
            S diff = std::abs(gh2[i*q] - gh[i]);
            l1_diff_abs += diff;
            l2_diff_abs += diff * diff;
            linf_diff_abs = std::max(linf_diff_abs, std::abs(gh2[i*q] - gh[i]));

            //Абсолютные нормы 
            l1_abs += std::abs(gh[i]);
            l2_abs += gh[i] * gh[i];
            linf_abs = std::max(linf_abs, std::abs(gh[i]));
        }

        
        Norms<Span> norms{
            .L_1_abs = l1_diff_abs,
            .L_2_abs = std::sqrt(l2_diff_abs),
            .L_inf_abs = linf_diff_abs,

            .L_1_rel = l1_diff_abs / l1_abs,
            .L_2_rel = std::sqrt(l2_diff_abs) / std::sqrt(l2_abs),
            .L_inf_rel = linf_diff_abs / linf_abs
        };
        return norms;
    }

    template<class Span>
    void write_norms_table(
                        std::ofstream& out, 
                        typename Span::Scalar h, 
                        std::tuple<std::string_view, std::string_view, Norms<Span>> const& method_step_norms, 
                        bool header
                    ){
        if (header == true)
            out << "method,h,y_step," "L_1_abs,L_2_abs,L_inf_abs,L_1_rel,L_2_rel,L_inf_rel\n";
        
        auto norms = std::get<2>(method_step_norms);

        out << std::setprecision(5) << std::scientific << std::get<0>(method_step_norms) << ',' << h << ',' << std::get<1>(method_step_norms) << ','
            << norms.L_1_abs << ',' << norms.L_2_abs << ',' << norms.L_inf_abs << ','
            << norms.L_1_rel << ',' << norms.L_2_rel << ',' << norms.L_inf_rel << '\n';
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