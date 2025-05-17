#include <cmath>
#include "odesolver.hpp"

using namespace std;

int main(){
    //ДУ
    auto rhs = [](double x, double y1, double y2){return y2 * cos(x) + y1 * sqrt(x + 1) + pow(x,2) - 1;};
    double q0 = 1.7, q1 = 2.0;
    double a = 0.0, b = 1.0;

    //Требуемая точность
    double tol = 1e-5;

    using KernelTEuler = ode::EulerStepper<ali_span<double>, decltype(rhs)>;
    using KernelTEulerRec = ode::EulerRecountStepper<ali_span<double>, decltype(rhs)>;
    using KernelTRK2 = ode::RK2Stepper<ali_span<double>, decltype(rhs)>;
    using KernelTRK4 = ode::RK4Stepper<ali_span<double>, decltype(rhs)>;
    using KernelTAdams = ode::AdamsStepper<ali_span<double>, decltype(rhs)>;

    //Создание ядрер интегрирования
    KernelTEuler euler;
    KernelTEulerRec euler_rec;
    KernelTRK2 rk2;
    KernelTRK4 rk4;
    KernelTAdams adams;

    //Результаты формата (x, y, y', success)
    auto Result_euler = ode::integrate_adaptive<101, ali_span<double>>(rhs, a, b, q0, q1, tol, euler);
    auto Result_euler_rec = ode::integrate_adaptive<101, ali_span<double>>(rhs, a, b, q0, q1, tol, euler_rec);
    auto Result_rk2 = ode::integrate_adaptive<101, ali_span<double>>(rhs, a, b, q0, q1, tol, rk2);
    auto Result_rk4 = ode::integrate_adaptive<5, ali_span<double>>(rhs, a, b, q0, q1, tol, rk4);
    auto Result_adams = ode::integrate_adaptive<11, ali_span<double>>(rhs, a, b, q0, q1, tol, adams);

    auto Freeze_euler = ode::integrate_freeze<25, ali_span<double>>(rhs, a, b, q0, q1, euler);


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
        make_pair(Result_rk2.y2(), "rk2_y2"),

        make_pair(Result_rk4.x(), "rk4_x"),
        make_pair(Result_rk4.y1(), "rk4_y1"), 
        make_pair(Result_rk4.y2(), "rk4_y2"),

        make_pair(Result_adams.x(), "adams_x"),
        make_pair(Result_adams.y1(), "adams_y1"), 
        make_pair(Result_adams.y2(), "adams_y2"),

        make_pair(Freeze_euler.x(), "feuler_x"),
        make_pair(Freeze_euler.y1(), "feuler_y1"), 
        make_pair(Freeze_euler.y2(), "feuler_y2")
    
    );
    cout << "success: " << Result_euler.success << " " << Result_euler.x().size() << " " << Result_euler_rec.success << " " << Result_euler_rec.x().size() << " " << Result_rk2.success << " " << Result_rk2.x().size() << " " << Result_rk4.success << " " << Result_rk4.x().size() << " " << Result_adams.x().size() << '\n';
    cout << Result_euler.x()[1] - Result_euler.x()[0] << " "  << Result_euler_rec.x()[1] - Result_euler_rec.x()[0] <<  " " << Result_rk2.x()[1] - Result_rk2.x()[0] <<  " " << Result_rk4.x()[1] - Result_rk4.x()[0] << " " << Result_adams.x()[1] - Result_adams.x()[0]<<'\n';
    system("/home/stonexis/PycharmProjects/PythonProject/.venv/bin/python3 plotter.py");
    










    return 0;
}