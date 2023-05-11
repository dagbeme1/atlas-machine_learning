#!/usr/bin/env python3
""" calculate the shape of a matrix using nested loops """
def matrix_shape(matrix):  # parameter(matrix)
        shape = []  # an empty list.stores dimensions of the matrix
        while isinstance(matrix, list):  
            shape.append(len(matrix))  # iteration of the outer loop, length
            matrix = matrix[0]
        return shape[::-1]  #  list in reverse order,matrix from outermost to innermost

