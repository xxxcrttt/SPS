#import utilities

#import importlib

#importlib.import_module('utilities')

#filename = sys.argv[1]

#plot_no_plot = ""
#if(len(sys.argv) > 2):
#    plot_no_plot  = sys.argv[2]

#x, y = load_file("train/" + str(filename))
from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import os
import sys
import numpy as np
from pprint import pprint
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import load_points_from_file, view_data_segments

def least_squares(x, y):
    ones = np.ones(x.shape)
    newx = np.column_stack((ones, x))
    a = np.linalg.inv(newx.T.dot(newx)).dot(newx.T).dot(y)
    return(a)

def poly_least_squares(x, y, degree):
    ones = np.ones(x.shape)
    newx = np.column_stack((ones, x))
    for i in range(2, degree + 1):
        newx = np.column_stack((newx, x**i))
    a = np.linalg.inv(newx.T.dot(newx)).dot(newx.T).dot(y)
    return(a)

def sumSquaredError(y, yLine):
    yTemp = yLine - y
    yTemp = np.square(yTemp)
    sumSquaredErrorValue = sum(yTemp)
    return(sumSquaredErrorValue)

def calculateY(ab, degree, testPointsX):
    for degrees in range(1, degree + 1):
        if degrees == 1:
            z = ab[0] + ab[degrees] * testPointsX**degrees
        else:
            z = ab[degrees] * testPointsX**degrees + z

    return(z)

testPoints = load_points_from_file(sys.argv[1])
# print(testPoints)

numberOfSegments = int(len(testPoints[0]) / 20)
# print(numberOfSegments)
plt.scatter(testPoints[0], testPoints[1])

degree = 5
bestDegree = 0
errorCounter = 0
for outer in range(numberOfSegments):
    minError = 100000000

    testPointsX = testPoints[0][outer * 20:outer * 20 + 20]
    testPointsY = testPoints[1][outer * 20:outer * 20 + 20]

    ab = poly_least_squares(testPointsX, testPointsY, 1)
    print('degrees', 1, 'y', ab)
    z = calculateY(ab, 1, testPointsX)
    linearError = sumSquaredError(testPointsY, z)
    print('degrees', 1, 'error', linearError)

    for degrees in range(2, degree + 1):

        ab = poly_least_squares(testPointsX, testPointsY, degrees)
        print('degrees', degrees, 'y', ab)

        z = calculateY(ab, degrees, testPointsX)

        error = sumSquaredError(testPointsY, z)
        print('degrees', degrees, 'error', error)

        if(error < minError):
            bestDegree = degrees
            minError = error
            print('Min error', minError)
            print('New best degree', bestDegree)

    if(linearError < minError):
        minError = linearError
        bestDegree = 1
    else:
        percentageDifference = abs((minError - linearError) / linearError)

        if(percentageDifference < 0.4):
            minError = linearError
            bestDegree = 1
    #('EPRCENT DIFFERENCE', percentageDifference)

    ab = poly_least_squares(testPointsX, testPointsY, bestDegree)

    z = calculateY(ab, bestDegree, testPointsX)

    print('Min error', minError)

    plt.plot(testPointsX, z, 'red')

    errorCounter = minError + errorCounter
    #error = sumSquaredError(testPointsY, z)
    #tempError = error + tempError

#print(tempError)
print('Total Error:', errorCounter)
plt.show()
