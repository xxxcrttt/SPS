
#!/usr/bin/env python
# coding: utf-8




import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()




name = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == '--plot':
    plotting = True
else:
    plotting = False

x, y = load_points_from_file(name)

# least square
def leastSquaresLinear(x_in, y_in):
    ones = np.ones(x_in.shape)
    x_e = np.column_stack((ones, x_in))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y_in)
    return (v[0], v[1])

def leastSquaresCubic(x_in, y_in):
    ones = np.ones(x_in.shape)
    x_e = np.column_stack((ones, x_in, (x_in)**2, (x_in)**3))
    v= np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y_in)
    return (v[0], v[1], v[2], v[3])

def leastSquaresSine(x_in, y_in):
    ones = np.ones(x_in.shape)
    x_e = np.column_stack((ones, np.sin(x_in)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y_in)
    return (v[0], v[1])


# plot
def linearPlot(x_in, a_in, b_in, ax):
    y_in = a_in + b_in*x_in
    if(plotting): ax.plot(x_in,y_in)

def cubicPlot(x_in, a_in, b_in, c_in, d_in, ax):
    y_in = a_in + b_in*x_in + c_in*x_in**2 + d_in*x_in**3
    if(plotting): ax.plot(x_in,y_in)

def sinePlot(x_in, a_in, b_in, ax):
    y_in = a_in + b_in*np.sin(x_in)
    ax.plot(x_in,y_in)

# error
def linearError(x_in, y_in, a_in, b_in):
    Y = a_in + b_in * x_in
    error = np.sum((Y - y_in) ** 2)
    return error

def cubicError(x_in, y_in, a_in, b_in, c_in, d_in):
    Y = a_in + b_in * x_in + c_in*x_in**2 + d_in*x_in**3
    error = np.sum((Y - y_in) ** 2)
    return error

def sineError(x_in, y_in, a_in, b_in):
    Y = a_in + b_in * np.sin(x_in)
    error = np.sum((Y - y_in) ** 2)
    return error

xs = [x[i:i + 20] for i in range (0, len(x), 20)]
ys = [y[i:i + 20] for i in range (0, len(y), 20)]

# choose best fit to plot
fig, ax = plt.subplots()
error = 0
for k in range(0, len(xs)):
    a1, b1 = leastSquaresLinear(xs[k], ys[k])
    a2, b2, c2, d2 = leastSquaresCubic(xs[k], ys[k])
    a3, b3 = leastSquaresSine(xs[k], ys[k])

    linError = linearError(xs[k], ys[k], a1, b1)
    cuError = cubicError(xs[k], ys[k], a2, b2, c2, d2)
    sinError = sineError(xs[k], ys[k], a3, b3)

    if (sinError < 1.1*linError) and (sinError < 1.1*cuError):
        sinePlot(xs[k], a3, b3, ax)
        error =  error + sinError

    elif (linError < 1.1*cuError):
        linearPlot(xs[k], a1, b1, ax)
        error =  error + linError

    else:
        cubicPlot(xs[k], a2, b2, c2, d2, ax)
        error =  error + cuError

if(plotting): view_data_segments(x, y)
print (error)
