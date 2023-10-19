#!/usr/bin/env python3
"""Contains the class BayesianOptimization."""

import numpy as np
from scipy.stats import norm

# Import the GaussianProcess class from a module named '2-gp'.
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.

    This class is used for optimizing a black-box function in
    a one-dimensional space using Bayesian optimization.
    It employs a Gaussian Process model for modeling the unknown
    function and the Expected Improvement (EI) acquisition
    function to select the next best sample location.
    Bayesian optimization is suitable for both minimization and
    maximization tasks, and this class can be used for both objectives.

    Attributes:
        f (function): The black-box function to be optimized.
        gp (GaussianProcess): The Gaussian Process model
        used to represent the unknown function.
        X_s (numpy.ndarray): A grid of potential sample
        points to select the next sample from.
        xsi (float): The exploration-exploitation factor used
        in the acquisition function.
        minimize (bool): A boolean value that determines whether
        optimization is for minimization (True) or maximization (False).

    Methods:
        __init__(self, f, X_init, Y_init, bounds, ac_samples,
        l=1, sigma_f=1, xsi=0.01, minimize=True):
            Initializes a BayesianOptimization instance.

        acquisition(self):
            Calculates the next best sample location using
            Expected Improvement (EI).

        optimize(self, iterations=100):
            Optimizes the black-box function using Bayesian optimization.

    Args:
        f (function): The black-box function to be optimized.
        X_init (numpy.ndarray): Inputs already sampled
        with the black-box function.
        Y_init (numpy.ndarray): Outputs of the black-box
        function for each input in X_init.
        bounds (tuple): Tuple of (min, max) representing
        the bounds of the optimization space.
        ac_samples (int): The number of samples to
        be analyzed during acquisition.
        l (float, optional): Length parameter for
        the kernel. Default is 1.
        sigma_f (float, optional): Standard deviation given to
        the output of the black-box function. Default is 1.
        xsi (float, optional): The exploration-exploitation
        factor for acquisition. Default is 0.01.
        minimize (bool, optional): True for minimizing,
        False for maximizing. Default is True.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize the BayesianOptimization object.

        Args:
            f (function): The black-box function to be optimized.
            X_init (numpy.ndarray): Inputs already sampled
            with the black-box function.
            Y_init (numpy.ndarray): Outputs of the black-box
            function for each input in X_init.
            bounds (tuple): Tuple of (min, max) representing the
            bounds of the optimization space.
            ac_samples (int): The number of samples to be
            analyzed during acquisition.
            l (float, optional): Length parameter for the kernel. Default is 1
            sigma_f (float, optional): Standard deviation given to the
            output of the black-box function. Default is 1.
            xsi (float, optional): Exploration-exploitation factor
            for acquisition. Default is 0.01.
            minimize (bool, optional): True for minimizing,
            False for maximizing. Default is True.
        """
        # Initialize the black-box function and Gaussian Process model.
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        # Create a grid of potential sample points.
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate the next best sample location using
        Expected Improvement (EI).

        Returns:
            X_next (numpy.ndarray): Next best sample point of shape (1,)
            EI (numpy.ndarray): Expected improvement of each potential
            sample of shape (ac_samples,)
        """
        X = self.gp.X
        mu_sample, _ = self.gp.predict(X)

        mu, sigma = self.gp.predict(self.X_s)

        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='warn'):
            if self.minimize:
                mu_sample_opt = np.amin(self.gp.Y)
                imp = (mu_sample_opt - mu - self.xsi).reshape(-1, 1)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                imp = (mu - mu_sample_opt - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[np.isclose(sigma, 0)] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI.reshape(-1)

    def optimize(self, iterations=100):
        """
        Optimize the black-box function using Bayesian Optimization.

        Args:
            iterations (int, optional): Maximum number of iterations to
            perform.Default is 100.

        Returns:
            X_opt (numpy.ndarray): Optimal point of shape (1,)
            Y_opt (numpy.ndarray): Optimal function value of shape (1,)
        """
        for i in range(0, iterations):
            X_next, EI = self.acquisition()

            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
