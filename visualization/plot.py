import matplotlib.pyplot as plt
import numpy as np


def plot_2d_fibra():
    res_fibra = np.load('2d_fibra.npy')
    print(res_fibra)
    FE_merod = np.load('3metod.npy')
    time = np.array(range(len(FE_merod)))
    plt.plot(time,FE_merod, color = 'tab:blue', label='скаляр')
    plt.plot(time[1:],res_fibra, color = 'tab:red', label='тензор')
    plt.title('FL метод ')
    plt.ylabel("Напряжение [мВ]",fontsize=16)
    plt.xlabel("время [мс]",fontsize=16)
    plt.legend(fontsize=16)
    plt.show()



def plot_2d():
    matrix_metod = np.loadtxt('matrix_2d.txt')
    FE_merod = np.load('3metod.npy')
    direct = np.load('1metod.npy')
    time = np.array(range(len(direct)))

    plt.subplot(2, 2, 1)
    plt.plot(time,direct, color = 'tab:blue', label='время: 70 сек')
    plt.title('Полное решение',fontsize=12)
    plt.ylabel("Напряжение [мВ]",fontsize=12)
    plt.xlabel("время [мс]",fontsize=12)
    plt.legend(fontsize=10)


    #plt.show()
    plt.subplot(2, 2, 2)
    plt.plot(time[1:600],matrix_metod * (-1), color = 'tab:orange', label='4 сек')
    plt.title('Дискретный метод',fontsize=12)
    plt.ylabel("Напряжение [мВ]",fontsize=12)
    plt.xlabel("время [мс]",fontsize=12)
    plt.legend(fontsize=10)

    # plt.show()
    plt.subplot(2, 2, 3)
    plt.plot(time,FE_merod,color = 'tab:red', label='время: 2 сек')
    plt.title('LF метод',fontsize=12)
    plt.ylabel("Напряжение [мВ]",fontsize=12)
    plt.xlabel("время [мс]",fontsize=12)
    plt.legend(fontsize=10)


    plt.subplot(2, 2, 4)
    plt.plot(time,FE_merod,color = 'tab:red', label='FL метод')
    plt.plot(time[1:600],matrix_metod * (-1), color = 'tab:orange', label='Дискретный метод')
    plt.plot(time,direct, color = 'tab:blue', label='Полное решение')
    plt.title('Все методы',fontsize=12)
    plt.ylabel("Напряжение [мВ]",fontsize=12)
    plt.xlabel("время [мс]",fontsize=12)
    plt.legend(fontsize=10)
    plt.show()


def plot_3d_delta_fibra():
    res = np.load('3metod_3d.npy')
    res_fibra = np.load('3d_fibra.npy')
    time = np.array(range(len(res)))

    plt.plot(time,res, color = 'tab:blue', label='скаляр')
    plt.plot(time,res_fibra, color = 'tab:red', label='тензор')
    plt.title('FL метод ')

    plt.ylabel("Напряжение [мВ]",fontsize=16)
    plt.xlabel("время [мс]",fontsize=16)
    plt.legend(fontsize=16)
    plt.show()


if __name__ == '__main__':
    # plot_3d_delta_fibra()
    # plot_2d()
    plot_2d_fibra()