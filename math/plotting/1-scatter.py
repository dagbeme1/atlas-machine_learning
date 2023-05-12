#!/usr/bin/env python3
import numpy as np  # imports the NumPy library and allows you to refer to it using the alias np
import matplotlib.pyplot as plt  # a plotting library for Python that imports the pyplot from matplotlib

mean = [69, 0]  # a list mean with two elements 69 and 0
cov = [[15, 8], [8, 15]]  # a 2x2 list cov that represents the covariance matrix
np.random.seed(5)
""" generates 2000 random samples from a multivariate normal distribution. T transposes the resulting array, so x and y are separate arrays representing the x and y """
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180  # adds 180 to every element in the y array

plt.scatter(x, y, color='m')  # creates a scatter plot using the scatter function from Matplotlib
plt.xlabel('Height (in)')  # x-axis is heights
plt.ylabel('Weight (lbs)')  # y-axis is weights
plt.title("Men's Height vs Weight")  # title of the scatter graph
plt.show()  # displays scatter graph
