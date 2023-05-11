#!/usr/bin/env python3
""" matrix addition of arrays  """


def add_arrays(arr1, arr2):  # defines a function named add_arrays with 2 para
    """ Two arrays add and returns """
    if len(arr1) != len(arr2):  # checks if the lengths of arr1 and arr2
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
