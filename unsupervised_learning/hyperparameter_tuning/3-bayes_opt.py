#!/usr/bin/env python3
"""
3-bayes_opt.py
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that instantiates a Bayesian optimization
    on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Define and initialize variables and methods

        Args:
            f: The function to optimize.
            X_init (numpy.ndarray): Inputs already sampled with
            the black-box function.
            Y_init (numpy.ndarray): Outputs of the black-box
            function for each input.
            bounds (tuple): A tuple (min, max) specifying the
            bounds of the optimization.
            ac_samples (int): The number of samples to use for acquisition.
            l (float, optional): Length parameter for the kernel.
            Default is 1.
            sigma_f (float, optional): Standard deviation given to the
            output of the black-box function. Default is 1.
            xsi (float, optional): Exploration-exploitation factor.
            Default is 0.01.
            minimize (bool, optional): True if optimizing for minimum,
            False for maximum. Default is True.
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize
