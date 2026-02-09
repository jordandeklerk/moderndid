"""GPU-native WLS and logistic regression via backend dispatch."""

import numpy as np

from .backend import get_backend


def cupy_wls(y, X, weights):
    """Weighted least squares via direct linear algebra.

    Parameters
    ----------
    y : ndarray
        Response vector of shape (n,).
    X : ndarray
        Design matrix of shape (n, k).
    weights : ndarray
        Non-negative weight vector of shape (n,).

    Returns
    -------
    beta : ndarray
        Coefficient vector of shape (k,).
    fitted : ndarray
        Fitted values X @ beta of shape (n,).
    """
    xp = get_backend()
    try:
        w = xp.asarray(weights, dtype=xp.float64)
        XtW = X.T * w
        beta = xp.linalg.solve(XtW @ X, XtW @ y)
        return beta, X @ beta
    except Exception as e:
        if xp is not np and "OutOfMemory" in type(e).__name__:
            raise MemoryError("GPU out of memory during WLS. Reduce batch size or use backend='numpy'.") from None
        raise


def cupy_logistic_irls(y, X, weights, max_iter=25, tol=1e-8):
    """Logistic regression via iteratively reweighted least squares.

    Parameters
    ----------
    y : ndarray
        Binary response vector of shape (n,).
    X : ndarray
        Design matrix of shape (n, k).
    weights : ndarray
        Non-negative weight vector of shape (n,).
    max_iter : int, default 25
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on max absolute parameter change.

    Returns
    -------
    beta : ndarray
        Coefficient vector of shape (k,).
    mu : ndarray
        Predicted probabilities of shape (n,).
    """
    xp = get_backend()
    try:
        beta = xp.zeros(X.shape[1], dtype=xp.float64)
        for _ in range(max_iter):
            eta = X @ beta
            mu = 1.0 / (1.0 + xp.exp(-eta))
            mu = xp.clip(mu, 1e-10, 1 - 1e-10)
            W = weights * mu * (1 - mu)
            z = eta + (y - mu) / (mu * (1 - mu))
            XtWX = X.T @ (W[:, None] * X)
            XtWz = X.T @ (W * z)
            beta_new = xp.linalg.solve(XtWX, XtWz)
            if xp.max(xp.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        mu = 1.0 / (1.0 + xp.exp(-(X @ beta)))
        mu = xp.clip(mu, 1e-10, 1 - 1e-10)
        return beta, mu
    except Exception as e:
        if xp is not np and "OutOfMemory" in type(e).__name__:
            raise MemoryError(
                "GPU out of memory during logistic IRLS. Reduce batch size or use backend='numpy'."
            ) from None
        raise
