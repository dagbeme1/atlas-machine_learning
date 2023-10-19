#!/usr/bin/env python3
"""
BayesianOptimization module

This module contains the `BayesianOptimization` class for performing Bayesian
optimization on a noiseless 1D Gaussian process.
"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian Optimization for Noiseless 1D Gaussian Process

    This class performs Bayesian optimization on a noiseless 1D Gaussian
    process. It is used to find the optimal input of a
    black-box function within specified bounds.

    Attributes:
        f (function): The black-box function to be optimized.
        X_init (numpy.ndarray): Inputs already sampled with
        the black-box function.
        Y_init (numpy.ndarray): Outputs of the black-box function
        for each input in X_init.
        bounds (tuple): The bounds of the optimization space (min, max).
        ac_samples (int): The number of samples to use for acquisition.
        l (float, optional): Length parameter for the kernel (default is 1).
        sigma_f (float, optional): Standard deviation given to the output
        of the black-box function (default is 1).
        xsi (float, optional): Exploration-exploitation factor for
        acquisition (default is 0.01).
        minimize (bool, optional): True for minimization,
        False for maximization (default is True).
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Constructor for BayesianOptimization.

        Args:
            f: black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1)
                representing the inputs already sampled with
                the black-box function
            Y_init: numpy.ndarray of shape (t, 1)
            representing the outputs
                of the black-box function for each input in X_init
            bounds: tuple of (min, max) representing the bounds
                of the space in which to look for the optimal point
            ac_samples: number of samples that should be analyzed
                during acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization
                should be performed for minimization (True) or
                maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
    Calculate the next best sample location using
    Expected Improvement (EI) method.

    This method calculates the next best sample location based on the Expected
    Improvement (EI) acquisition function. It estimates the optimal
    point within the specified bounds.

    Returns:
        X_next (numpy.ndarray): The next best sample point with shape (1,).
        EI (numpy.ndarray): The Expected Improvement for each potential
        sample with shape (ac_samples,).
        """
        X = self.gp.X
        mu_sample, _ = self.gp.predict(X)

        mu, sigma = self.gp.predict(self.X_s)

        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='warn'):
            if self.minimize is True:
                mu_sample_opt = np.amin(self.gp.Y)
                imp = (mu_sample_opt - mu - self.xsi).reshape(-1, 1)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                imp = (mu - mu_sample_opt - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI.reshape(-1)
