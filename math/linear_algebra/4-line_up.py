#!/usr/bin/env python3
""" matrix addition of arrays  """


def add_arrays(arr1, arr2):  # defines a function named add_arrays with 2 parameters arr1 and arr2
    """ Two arrays add and returns """
    if len(arr1) != len(arr2):  # checks if the lengths of arr1 and arr2 are not equal. If they have different lengths, and cannot add them
        return None

    result = []  # store the element-wise addition of the arrays
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])  #  iterates over the indices of arr1 (or arr2 since they have the same length) using the range function

    return result
