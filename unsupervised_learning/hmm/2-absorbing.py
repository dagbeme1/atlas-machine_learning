#!/usr/bin/env python3
"""
the function def absorbing(P): that determines
if a markov chain is absorbing
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state
    after a specified number of iterations.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n), where P[i, j]
            is the probability of transitioning from state i to state j.
        s (numpy.ndarray): Initial state probabilities of shape (1, n).
        t (int): Number of iterations.

    Returns:
        numpy.ndarray: Probability of being in
        a specific state after t iterations,
        or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]

    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), np.ones(n)):
        return None

    if not isinstance(
            s,
            np.ndarray) or s.shape != (
            1,
            n) or not np.allclose(
                np.sum(
                    s,
                    axis=1),
            [1]):
        return None

    if not isinstance(t, int) or t < 0:
        return None

    Pt = np.linalg.matrix_power(P, t)
    Ps = np.matmul(s, Pt)

    return Ps


def regular(P):
    """
    Determines the steady-state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n), where P[i, j]
            is the probability of transitioning from state i to state j.

    Returns:
        numpy.ndarray: Steady-state probabilities of shape (1, n),
        or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]

    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), np.ones(n)):
        return None

    s = np.full((1, n), 1 / n)
    Pk = np.copy(P)

    while True:
        Pk = np.matmul(Pk, P)
        s_new = np.matmul(s, P)

        if np.allclose(s, s_new):
            return s

        s = s_new


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n), where P[i, j]
            is the probability of transitioning from state i to state j.

    Returns:
        bool: True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]

    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), np.ones(n)):
        return False

    if np.all(np.diag(P) == 1):
        return True

    return route_check(P, n)


def route_check(P, n):
    absorbing_states = np.where(np.diag(P) == 1)
    rows = P[absorbing_states[0]]
    account = np.sum(rows, axis=0)

    for i in range(n):
        row_check = P[i] != 0
        intersection = account * row_check

        if (intersection == 1).any():
            account[i] = 1

    return account.all()
