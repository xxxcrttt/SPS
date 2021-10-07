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
    plt.scatter(xs, ys, s=1, c=colour)
    plt.show()

# print("xs:",xs, "ys:", ys)

#Least Squares(Calculate slope and y-intercept)
def find_optimal_weights(x,y):

    #x_e = np.vander(x,p, True)
    v = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)#dot means matrix multiplication
    return v

# a_1, b_1 = least_squares(xs,ys)
# print("a:",a_1, "b:", b_1)



""""def create_line(xs,ys):
    orig_x = xs
    weights = least_squares(xs,ys,2)
    ones = np.ones(xs.shape)
    xs = np.column_stack((ones, xs, xs**2 ))
    xs_min = xs.min()
    xs_max = xs.max()
    ys_min = weight + b_1 * xs_min
    ys_max = a_1 + b_1 * xs_max
    predicted_y = np.dot(xs, weights)
    print(predicted_y)
    print(square_error(ys, predicted_y))
    plt.scatter(orig_x, ys)
    plt.plot(orig_x, predicted_y)
    #ax.scatter(xs,ys,s=200)
    #ax.plot([xs_min, xs_max],[ys_min,ys_max],'r-',lw = 2)"""

seg_err= [] #error for each line
seg_min= [] #min error for each line
seg_err1= [] #error for each line
seg_min1= [] #min error for each line

def fit (xs,ys,degree):
    new_xs = create_polynomial(xs, degree)#call create_polynominal function
    weights = find_optimal_weights(new_xs, ys)#call find_optimal_weights function
    pred_y = new_xs.dot(weights)#It is the same as y_hat that we define in square_error function, which is the predicted y.
    #     print(pred_y)
    err = square_error(ys,pred_y)#the sum error between predicted y and real y
    seg_err.append(err)
    seg_min=min(seg_err)
    #     print(err)
    new_xs1 =  create_sin(xs)
    weights1 = find_optimal_weights(new_xs1, ys)
    pred_y1 = weights1[0] + weights1[1] * np.sin(xs)
    err1 = square_error(ys,pred_y1)#the sum error between predicted y and real y
    seg_err1.append(err1)
    seg_min1=min(seg_err1)

    return (pred_y, err, pred_y1, err1)

#create polynominal matrix
def create_polynomial(xs, degree):
    """
        param xs: np array
        """
    ones = np.ones(xs.shape)
    original_x = xs
    xs = np.column_stack((ones, xs))
    for d in range(2, degree+1):
        xs = np.c_[xs, original_x**d] #combine an array in the matrix vertically
    return xs

def create_sin(xs):
    """
        param xs: np array
        """
    ones = np.ones(xs.shape)
    original_x = xs
    xs = np.column_stack((ones, np.sin(xs)))
    return xs



def square_error(y,y_hat):
    return np.sum((y_hat - y) ** 2)

#plot
fig, ax = plt.subplots()
xs, ys = load_points_from_file(sys.argv[1])
#deal with files that contain 20n(n>1) points
xi = []#xi contains 20 points
yi = []#yi contains 20 points

for n in range(int(len(xs) / 20)):
    seg_xs = []
    seg_ys = []
    for h in range(n*20, n*20 + 20):
        seg_xs.append(xs[h])
        seg_ys.append(ys[h])
    xi.append(seg_xs)
    yi.append(seg_ys)


#choose the best degree for the points
o = 10#highest degree
def choose_degree(xs, ys):
    assert(len(xs) == len(ys) == 20)#it is like the try-except statement in java
    smallest_err = float("inf")#infinity
    most_accurate_y = None
    for i in range(1, o+1):
        (pred_y, err, pred_y1, err1) = fit(xs,ys,i)
        if err < smallest_err:
            smallest_err = err
            most_accurate_y = pred_y
            if err1 < smallest_err:
                smallest_err = err1
                most_accurate_y = pred_y1

    plt.plot(xs, most_accurate_y)

plt.scatter(xs, ys)

#main function
for m in range(len(xi)):
    choose_degree(np.array(xi[m]), np.array(yi[m]))
if len(sys.argv) == 3 and sys.argv[2] == '--plot':
    plt.show()

for i in range(0, len(seg_err), o):
    seg_min.append(min(seg_err[i:i+o]))
for i in range(0, len(seg_err1), o):
    seg_min1.append(min(seg_err1[i:i+o]))

seg_min2 = [0 for i in range(int(len(xs) / 20))]
for n in range(int(len(xs) / 20)):
    if seg_min[n] < seg_min1[n]:
        seg_min2[n] = seg_min[n]
    else:
        seg_min2[n] = seg_min1[n]

print(sum(seg_min2))
