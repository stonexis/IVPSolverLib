import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


# def plot_graphs(filename, nrows, ncols):
#     with open(filename, "r") as f:
#         data = json.load(f)

#     sns.set_theme(style="whitegrid")

#     methods = [
#         ("Euler", data["euler_x"], data["euler_y1"], data["euler_y2"]),
#         ("Euler_rec", data["euler_rec_x"], data["euler_rec_y1"], data["euler_rec_y2"]),
#         ("RK2",  data["rk2_x"], data["rk2_y1"], data["rk2_y2"]),
#         ("RK4",  data["rk4_x"], data["rk4_y1"], data["rk4_y2"]),
#         ("Adams",  data["adams_x"], data["adams_y1"], data["adams_y2"]),
#         ("FEuler", data["feuler_x"], data["feuler_y1"], data["feuler_y2"])
#     ]

#     fig, axes = plt.subplots(nrows, ncols,
#                              figsize=(ncols * 5, nrows * 4),
#                              sharey=True)
#     # превращаем axes в плоский список для удобства
#     axes = axes.flatten()

#     for ax, (title, x, y1, y2) in zip(axes, methods):
#         sns.lineplot(x=x, y=y1, label="y1", ax=ax)
#         sns.lineplot(x=x, y=y2, label="y2", ax=ax)
#         ax.set_title(title)
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.legend()
#         ax.xaxis.set_minor_locator(AutoMinorLocator(4))  # разобьёт каждый интервал между мажорными делениями на 4 части
#         ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#         ax.minorticks_on()  # включить отображение минорных меток
#         ax.grid(which='minor', linewidth=0.5, alpha=0.6)
        
#     #Удаление лишних осей, если есть
#     for ax in axes[len(methods):]:
#         fig.delaxes(ax)

#     plt.tight_layout()
#     plt.show()


# plot_graphs("data.json", nrows=2, ncols=3)

def plot_graphs(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid")

    euler_x = data["euler_x"]
    euler_rec_x = data["euler_rec_x"]
    rk2_x = data["rk2_x"]
    rk4_x = data["rk4_x"]
    adams_x = data["adams_x"]

    euler_y1 = data["euler_y1"]
    euler_rec_y1 = data["euler_rec_y1"]
    rk2_y1 = data["rk2_y1"]
    rk4_y1 = data["rk4_y1"]
    adams_y1 = data["adams_y1"]

    euler_y2 = data["euler_y2"]
    euler_rec_y2 = data["euler_rec_y2"]
    rk2_y2 = data["rk2_y2"]
    rk4_y2 = data["rk4_y2"]
    adams_y2 = data["adams_y2"]
    
    # Установка темы и стиля
    sns.set_theme(style="whitegrid", palette="muted", font="serif", font_scale=1.2)

    # Построение графика
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    sns.lineplot(x=euler_x, y=euler_y1, ax=axes[0], label='euler')
    sns.lineplot(x=euler_rec_x, y=euler_rec_y1, ax=axes[0], label='euler_rec')
    sns.lineplot(x=rk2_x, y=rk2_y1, ax=axes[0], label='rk2')
    sns.lineplot(x=rk4_x, y=rk4_y1, ax=axes[0], label='rk4')
    sns.lineplot(x=adams_x, y=adams_y1, ax=axes[0], label='adams')

    sns.lineplot(x=euler_x, y=euler_y2, ax=axes[1], label='euler')
    sns.lineplot(x=euler_rec_x, y=euler_rec_y2, ax=axes[1], label='euler_rec')
    sns.lineplot(x=rk2_x, y=rk2_y2, ax=axes[1], label='rk2')
    sns.lineplot(x=rk4_x, y=rk4_y2, ax=axes[1], label='rk4')
    sns.lineplot(x=adams_x, y=adams_y2, ax=axes[1], label='adams')


    axes[0].set_title("y1")
    axes[1].set_title("y2")
    # Настройки графика
    plt.xlabel("X", fontsize=14)
    plt.ylabel("value", fontsize=14)
    

    axes[0].xaxis.set_minor_locator(AutoMinorLocator(4))  # разобьёт каждый интервал между мажорными делениями на 4 части
    axes[0].yaxis.set_minor_locator(AutoMinorLocator(4))
    axes[0].minorticks_on()  # включить отображение минорных меток
    axes[0].grid(which='minor', linewidth=0.5, alpha=0.6)

    axes[1].xaxis.set_minor_locator(AutoMinorLocator(4))  # разобьёт каждый интервал между мажорными делениями на 4 части
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(4))
    axes[1].minorticks_on()  # включить отображение минорных меток
    axes[1].grid(which='minor', linewidth=0.5, alpha=0.6)

    plt.tight_layout()

    # Отображение графика
    plt.show()


plot_graphs("data.json")