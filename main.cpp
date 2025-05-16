#include <cmath>
#include "odesolver.hpp"

using namespace std;

int main(){
    //ДУ
    auto rhs = [](double x, double y1, double y2){return y2 * cos(x) + y1 * sqrt(x + 1) + pow(x,2) - 1;};
    double q0 = 2.0, q1 = 2.0;
    double a = 0.0, b = 1.0;

    //Требуемая точность
    double tol = 1e-5;

    using KernelTEuler = ode::EulerStepper<ali_span<double>, decltype(rhs)>;
    using KernelTEulerRec = ode::EulerRecountStepper<ali_span<double>, decltype(rhs)>;
    using KernelTRK2 = ode::RK2Stepper<ali_span<double>, decltype(rhs)>;

    //Создание ядрер интегрирования
    KernelTEuler euler;
    KernelTEulerRec euler_rec;
    KernelTRK2 rk2;
    //Результаты формата (x, y, y', success)
    auto Result_euler = ode::integrate_adaptive<10000, ali_span<double>>(rhs, a, b, q0, q1, tol, euler);
    auto Result_euler_rec = ode::integrate_adaptive<10000, ali_span<double>>(rhs, a, b, q0, q1, tol, euler_rec);
    auto Result_rk2 = ode::integrate_adaptive<10000, ali_span<double>>(rhs, a, b, q0, q1, tol, rk2);
    
    write_data_to_json_file(
        "data.json", 
        make_pair(Result_euler.x(),  "euler_x"), 
        make_pair(Result_euler.y1(), "euler_y1"), 
        make_pair(Result_euler.y2(), "euler_y2"),
        
        make_pair(Result_euler_rec.x(), "euler_rec_x"),
        make_pair(Result_euler_rec.y1(), "euler_rec_y1"), 
        make_pair(Result_euler_rec.y2(), "euler_rec_y2"),

        make_pair(Result_rk2.x(), "rk2_x"),
        make_pair(Result_rk2.y1(), "rk2_y1"), 
        make_pair(Result_rk2.y2(), "rk2_y2")
    
    );

    //system("/home/stonexis/PycharmProjects/PythonProject/.venv/bin/python3 plotter.py");
    










    return 0;
}