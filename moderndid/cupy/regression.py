"""GPU-native WLS and logistic regression via backend dispatch."""

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from .backend import get_backend, to_numpy


def cupy_wls(y, X, weights, return_condition_number=False):
    """Fit weighted least squares through a QR factorization of the weighted design.

    The rows of the design are scaled by the square roots of the weights and factored on the device.
    On the host, an SVD-based least squares solve of the small triangular factor treats singular values
    at or below 1e-15 times the largest as zero. A rank-deficient design therefore gets the minimum-norm
    coefficients instead of an error or non-finite values.

    Parameters
    ----------
    y : ndarray
        Response vector of shape (n,).
    X : ndarray
        Design matrix of shape (n, k).
    weights : ndarray
        Non-negative weight vector of shape (n,).
    return_condition_number : bool, default=False
        Whether to also return the condition number of the weighted design.

    Returns
    -------
    beta : ndarray
        Coefficient vector of shape (k,).
    fitted : ndarray
        Fitted values X @ beta of shape (n,).
    condition_number : float
        Ratio of the largest to the smallest singular value of the weighted design. Only returned when
        ``return_condition_number=True``.
    """
    xp = get_backend()
    try:
        w = xp.asarray(weights, dtype=xp.float64)
        r, effects = _reduced_system(xp, X, w, y)
        beta = xp.asarray(np.linalg.lstsq(r, effects, rcond=1e-15)[0])
        fitted = X @ beta
        if return_condition_number:
            # Since Q has orthonormal columns, R has the singular values of the weighted design.
            return beta, fitted, float(np.linalg.cond(r))
        return beta, fitted
    except Exception as e:
        if xp is not np and "OutOfMemory" in type(e).__name__:
            raise MemoryError("GPU out of memory during WLS. Reduce batch size or use backend='numpy'.") from None
        raise


def cupy_logistic_irls(y, X, weights, max_iter=100, tol=1e-8, return_converged=False):
    r"""Fit a weighted logistic regression by iteratively reweighted least squares.

    The fit starts from the probabilities :math:`(y + 0.5) / 2`. With the linear predictor
    :math:`\eta = X \beta`, the frequency weights :math:`w`, and the fitted probabilities :math:`\mu`
    kept machine precision away from zero and one, each iteration regresses the working response
    :math:`z` on the design with weights :math:`W` given by

    .. math::

        z = \eta + \frac{y - \mu}{\mu (1 - \mu)}, \qquad W = w \mu (1 - \mu).

    The fit stops once the weighted binomial deviance changes by at most ``tol`` between iterations.

    Each regression factors the weighted design as :func:`cupy_wls` does but keeps singular values down
    to machine precision. Since the returned coefficients solve the last regression with the cutoff of
    :func:`cupy_wls`, a rank-deficient design gets the minimum-norm coefficients.

    Separated outcomes leave the coefficients unidentified. An iteration whose fitted probabilities all
    lie within 1e-8 of the outcomes therefore warns with ``PerfectSeparationWarning``. Stopping at
    ``max_iter`` raises no warning. The caller reads ``converged`` and words its own warning.

    Parameters
    ----------
    y : ndarray
        Binary response vector of shape (n,).
    X : ndarray
        Design matrix of shape (n, k).
    weights : ndarray
        Non-negative frequency weight vector of shape (n,).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-8
        Tolerance on the absolute change in the deviance between iterations.
    return_converged : bool, default=False
        Whether to also return the convergence flag.

    Returns
    -------
    beta : ndarray
        Coefficient vector of shape (k,).
    mu : ndarray
        Fitted probabilities of shape (n,).
    converged : bool
        Whether the deviance settled within ``max_iter`` iterations. Only returned when
        ``return_converged=True``.
    """
    xp = get_backend()
    eps = np.finfo(np.float64).eps
    try:
        beta = xp.zeros(X.shape[1], dtype=xp.float64)
        mu = (y + 0.5) / 2
        clipped = xp.clip(mu, eps, 1 - eps)
        lin_pred = xp.log(clipped / (1 - clipped))
        deviance = _binomial_deviance(xp, y, mu, weights)
        converged = False
        for _ in range(max_iter):
            clipped = xp.clip(mu, eps, 1 - eps)
            variance = clipped * (1 - clipped)
            r, effects = _reduced_system(xp, X, weights * variance, lin_pred + (y - mu) / variance)
            beta = xp.asarray(np.linalg.lstsq(r, effects, rcond=-1)[0])
            lin_pred = X @ beta
            mu = 1 / (1 + xp.exp(-lin_pred))
            new_deviance = _binomial_deviance(xp, y, mu, weights)
            if bool(xp.all(xp.abs(mu - y) <= 1e-8)):
                warnings.warn(
                    "Perfect separation or prediction detected, parameter may not be identified",
                    PerfectSeparationWarning,
                    stacklevel=2,
                )
            converged = abs(new_deviance - deviance) <= tol
            deviance = new_deviance
            if converged:
                break
        # Since the iterations keep rounding-level singular values of a collinear design as the CPU fit does,
        # the coefficients solve the last system again with the 1e-15 cutoff of cupy_wls.
        if max_iter > 0:
            beta = xp.asarray(np.linalg.lstsq(r, effects, rcond=1e-15)[0])
        mu = 1 / (1 + xp.exp(-(X @ beta)))
        if return_converged:
            return beta, mu, converged
        return beta, mu
    except Exception as e:
        if xp is not np and "OutOfMemory" in type(e).__name__:
            raise MemoryError(
                "GPU out of memory during logistic IRLS. Reduce batch size or use backend='numpy'."
            ) from None
        raise


def _reduced_system(xp, X, weights, response):
    """Reduce a weighted least squares problem to its triangular factor and rotated response on the host."""
    root = xp.sqrt(weights)
    q, r = xp.linalg.qr(root[:, None] * X)
    return to_numpy(r), to_numpy(q.T @ (root * response))


def _binomial_deviance(xp, y, mu, weights):
    """Return the weighted binomial deviance with each log ratio floored at machine precision."""
    eps = np.finfo(np.float64).eps
    # Since a fitted probability can round to exactly zero or one, the offsets keep each ratio finite.
    success = xp.maximum(y / (mu + 1e-20), eps)
    failure = xp.maximum((1 - y) / (1 - mu + 1e-20), eps)
    return float(xp.sum(2 * weights * (y * xp.log(success) + (1 - y) * xp.log(failure))))
