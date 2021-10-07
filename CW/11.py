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

def view_data_segments(xs, ys, xx, yy, n):
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
    #for n in range(n):
        #plt.plot(xx[0][n], yy[n][0], color = 'r')
    plt.plot(xs, ys)
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def least_squares_linear(xi, yi):
    #Coloumn for ones
    #xi = list(xi)
    n = 1
    xs = []
    for i in range(n):
        xs.append(xi[i]*i)
    #gen = (xi**i for i in range(n+1))
    #for val in gen:
        #xs.append(val)

    ones = np.ones(xi.shape)
    X = np.column_stack((ones, xi))
    #Y = np.matrix([yi])
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)

    #Return  yhat value
    yhat = np.dot(X, A)
    return np.array(yhat)

def least_squares_poly(xi, yi):
    #Coloumn for ones
    #xi = list(xi)
    n = 3
    xs = []
    for i in range(n):
        xs.append(xi[i]**i)

    ones = np.ones(len(xi))
    X = np.column_stack((ones, xi))
    #Y = np.matrix([yi])
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)

    #Return  yhat value
    yhat = np.dot(X, A)
    return np.array(yhat)

def least_squares_unknown_sin(xi, yi):
    #Coloumn for ones
    #y = (xi**i for i in range(n+1))
    #for val in y:
    #    xs.append(val)

    ones = np.ones(len(xi))
    X = np.column_stack((ones, np.sin(xi)))
    #Y = np.matrix([yi])
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yi)
    #Return  yhat value
    yhat = np.dot(X, A)
    return np.array(yhat)

def sum_squared_error(yhat,y):
    #Calculate the sum squared error
    sumerr = np.sum((yhat-y)**2)
    return sumerr

if __name__ == "__main__" :
    fig, ax = plt.subplots()
    xs, ys = load_points_from_file(sys.argv[1])

    numLineSegs =len(xs)//20
    xsp = np.split(xs, numLineSegs)
    ysp = np.split(ys, numLineSegs)

    sumerr = 0
    yy = []

    for i in range(len(xsp)):
        yl = least_squares_linear(xsp[i], ysp[i])
        yp = least_squares_poly(xsp[i], ysp[i])
        yu = least_squares_unknown_sin(xsp[i], ysp[i])

        yle = sum_squared_error(yl, ysp[i])
        ype = sum_squared_error(yp, ysp[i])
        yue = sum_squared_error(yu, ysp[i])

        if (yle < ype and yle < yue):
            yy.append(yl)
        elif (ype < yle and ype < yue):
            yy.append(yp)
        else:
            yy.append(yu)

            sumerr = sumerr + min(yle, ype, yue)

    print(sumerr)

    xx = []
    xx.append(xsp)
    if (len(sys.argv) == 3):
        if (sys.argv[2] == '--plot'):
            view_data_segments(xs, ys, xx, yy, numLineSegs)

#view_data_segments(load_points_from_file(sys.argv[1])[0], load_points_from_file(sys.argv[1])[1])
#least_squares_linear(load_points_from_file(sys.argv[1])[0], load_points_from_file(sys.argv[1])[1])
