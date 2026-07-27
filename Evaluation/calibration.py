"""Post-hoc probability calibration, by two methods.

    temperature:  p_scaled = sigmoid(logit(p) / T)
    platt:        p_scaled = sigmoid(a * logit(p) + b)

Temperature scaling has one parameter and divides the logit, so it can only pull
scores toward one half. That suits a task whose base rate sits near one half and
whose scores are over-spread. Where the base rate sits elsewhere and the scores
are already centered on it, dividing the logit introduces a bias no single
parameter can undo, and the intercept in Platt scaling is what supplies it.

Both are fit by minimizing Bernoulli negative log likelihood against observed
correctness. Neither changes the ranking, only the scaled probability.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-6
T_BOUNDS = (0.05, 10.0)
PLATT_START = (1.0, 0.0)


def confidence_logit(confidence):
    p = np.clip(np.asarray(confidence, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def scaled_confidence(confidence, temperature):
    z = confidence_logit(confidence) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def temperature_nll(temperature, logits, labels):
    a = logits / temperature
    return float(np.mean(np.logaddexp(0.0, a) - labels * a))


def fit_temperature(confidence, labels):
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0 or labels.min() == labels.max():
        return 1.0
    logits = confidence_logit(confidence)
    result = minimize_scalar(
        temperature_nll, bounds=T_BOUNDS, args=(logits, labels), method="bounded"
    )
    return float(result.x)


def platt_nll(weights, logits, labels):
    a = weights[0] * logits + weights[1]
    return float(np.mean(np.logaddexp(0.0, a) - labels * a))


def platt_gradient(weights, logits, labels):
    """Exact gradient of platt_nll. Supplied rather than estimated, so the fit is
    the same to machine precision on any platform."""
    a = weights[0] * logits + weights[1]
    residual = 1.0 / (1.0 + np.exp(-a)) - labels
    return np.array([float(np.mean(residual * logits)), float(np.mean(residual))])


def platt_confidence(confidence, slope, intercept):
    z = slope * confidence_logit(confidence) + intercept
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(confidence, labels):
    """Slope and intercept by Bernoulli negative log likelihood. The objective is
    convex in both, so the optimum is unique and the starting point cannot change
    the answer. Identity parameters when the labels carry no signal to fit."""
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0 or labels.min() == labels.max():
        return 1.0, 0.0
    logits = confidence_logit(confidence)
    result = minimize(platt_nll, PLATT_START, jac=platt_gradient,
                      args=(logits, labels), method="BFGS")
    return float(result.x[0]), float(result.x[1])