import os, sys, numpy as np, utilities as utilities
from matplotlib import pyplot as plt

def draw_graph(ax, xs, ys, ys_hat):
    ax.scatter(xs, ys)
    ax.plot(xs,ys_hat)

def square_error(ys, ys_hat):
    return np.sum((ys - ys_hat) ** 2)

def power(x, y):
    return 1 if y == 0 else (x * power(x, y - 1))

def least_squares(xs, ys, p):
    xse = np.ones(xs.shape)
    n = 1
    ys_hat = 0
    while n <= p:
        xd = np.power(xs, n)
        xse = np.column_stack((xse, xd))
        n += 1
    rev = np.linalg.inv(xse.T@xse)@xse.T@ys
    for i in range(0, p + 1): ys_hat += (rev[i] * power(xs,i))
    return(square_error(ys, ys_hat), ys_hat)

def main(input):
    raw_xs, raw_ys = utilities.load_points_from_file('train_data/'+input)
    fig, ax = plt.subplots()
    n = len(raw_xs) // 20
    sum_error = 0
    while n > 0:
        xs = raw_xs [(n - 1) * 20 : n * 20]
        ys = raw_ys [(n - 1) * 20 : n * 20]
        se_array, ys_hat_array, index  = [],[],[]
        for i in range (1,10):
            se_single, ys_hat_single = least_squares(xs,ys, i)
            se_array.append(se_single)
            ys_hat_array.append(ys_hat_single) 
        # 在se_array中找最小的error的index       
        index = np.argmin(se_array)
        # 用这个 index 去找ys_hat
        ys_hat,sum_error = ys_hat_array[index], se_array[index]
        draw_graph(ax, xs, ys, ys_hat)
        n -= 1
    print('SE:',sum_error)
    plt.show() 

main('basic_1.csv')
main('basic_2.csv')
main('basic_3.csv')
main('basic_4.csv')
main('basic_5.csv')
main('adv_1.csv')
main('adv_2.csv')
main('adv_3.csv')
main('noise_1.csv')
main('noise_2.csv')
main('noise_3.csv')