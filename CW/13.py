from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utilities

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

    filename = load_points_from_file(sys.argv[1])
    #print(filename)
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

    #Change the colour for each segment
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])

    #Plot line graphs & scatter graphs for each segment
    #least_squares_linear(load_points_from_file(sys.argv[1])[0], load_points_from_file(sys.argv[1])[1])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def least_squares_linear(xi, yi):
    ones = np.ones(xi.shape)
    X = np.column_stack((ones, xi))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)
    return (A)

#up to order 5:
def least_squares_poly(xi, yi, degree):
    ones = np.ones(x.shape)
    X = np.column_stack((ones, xi))
    for i in range(2, degree + 1):
        X = np.column_stack((X, xi**i))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)
    return(A)

def least_squares_unknown_sin(xi, yi):
    ones = np.ones(xi.shape)
    X = np.column_stack((ones, np.sin(xi)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)
    return (A)

#calculate sum squared error
def sum_squared_error(yhat, y):
    sumerror = np.sum((yhat - y ) ** 2)
    return sumerror

def calculateY(ab, degree, xs):
    for degrees in range(1, degree + 1):
        if degrees == 1:
            z = ab[0] + ab[degrees] * xs**degrees
        else:
            z = ab[degrees] * xs**degrees + z

    return(z)

numberOfSegments = int(len(xs[0]) / 20)
# print(numberOfSegments)
plt.scatter(xs[0], xs[1])

degree = 5

for outer in range(numberOfSegments):
    xs = [x[i:i + 20] for i in range (0, len(x), 20)]
    ys = [y[i:i + 20] for i in range (0, len(y), 20)]

    ab = least_squares_poly(xs, ys, 1)
    z = calculateY(ab, 1, xs)
    linearError = sum_squared_error(xs, z)

    for degrees in range(2, degree + 1):

        ab = least_squares_poly(xs, ys, degrees)
        z  = calculateY(ab, degrees, xs)
        sumerror = sum_squared_error(ys, z)

        if (linerr <  )


#error
def linearerror (xi, yi, ai, bi):
    Y = ai + bi * xi
    sumerror = np.sum((Y - yi) ** 2)
    return sumerror

def polyerror (xi, yi, ai, bi, ci, di, ei):
    Y = ai + bi * xi + ci * xi**2 + di * xi**3 + ei * xi**4
    sumerror = np.sum((Y - yi) ** 2)
    return sumerror

def sinerror(xi, yi, ai, bi):
    Y = ai + bi * np.sin(xi)
    sumerror = np.sum((Y - yi) ** 2)
    return sumerror


#plot
name = sys.argv[1]
if len(sys.argv) == 3 and sys.argv[2] == '--plot':
    plotting = True
else :
    plotting = False

x, y = load_points_from_file(name)

xs = [x[i:i + 20] for i in range (0, len(x), 20)]
ys = [y[i:i + 20] for i in range (0, len(y), 20)]


def linearplot(xi, ai, bi, ax):
    yi = ai + bi * xi
    if(plotting): ax.plot(xi, yi)

def polyplot(xi, ai, bi, ci, di, ei, ax):
    yi = ai + bi * xi + ci * xi**2 + di * xi**3 + ei * xi**4
    if(plotting): ax.plot(xi, yi)

def sinplot(xi, ai, bi, ax):
    yi = ai + bi * np.sin(xi)
    ax.plot(xi, yi)

#choose best to fit
fig, ax = plt.subplots()
sumerror = 0

for i in range(0, len(xs)):

    a1, b1 = least_squares_linear(xs[i], ys[i])
    a2, b2, c2, d2, e2 = least_squares_poly(xs[i], ys[i])
    a3, b3 = least_squares_unknown_sin(xs[i], ys[i])

    linerr = linearerror(xs[i], ys[i], a1, b1)
    polyerr = polyerror(xs[i], ys[i], a2, b2, c2, d2, e2)
    sinerr = sinerror(xs[i], ys[i], a3, b3)

    if (sinerr < linerr and sinerr < polyerr):
        sinplot(xs[i], a3, b3, ax)
        sumerror = sumerror + sinerr
    elif (linerr < polyerr and linerr < sinerr):
        linearplot(xs[i], a1, b1, ax)
        sumerror = sumerror + linerr
    else:
        polyplot(xs[i], a2, b2, c2, d2, e2, ax)
        sumerror = sumerror + polyerr

if(plotting): view_data_segments(x, y)
print(sumerror)
