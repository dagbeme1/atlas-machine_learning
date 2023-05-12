#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Plot the line graph
plt.xlim(0, 10)  # set the x-axis range from 0 to 10
plt.plot(y, color='r')  # 'r' represents a solid red line
plt.show()  # Display the graph
