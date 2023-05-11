#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = [] # Creates an empty list called the_middle to store the 2D matrix containing the 3rd and 4th columns
for row in matrix:  # Starts a for loop that iterates over each row in the matrix
        the_middle.append(row[2:4]) # Extracts elements from the 3rd and 4th columns of the current row and adds them
print("The middle columns of the matrix are: {}".format(the_middle))
