"""Post-hoc probability calibration.

    temperature:  p = sigmoid(logit(p) / T)
    platt:        p = sigmoid(a * logit(p) + b)

Temperature has one parameter and can only pull scores toward one half. Platt
adds the intercept a base rate away from one half needs. Both fit Bernoulli
negative log likelihood, and neither changes the ranking.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

EPS = 1e-6
T_BOUNDS = (0.05, 10.0)
PLATT_START = (1.0, 0.0)


def confidence_logit(confidence):
    p = np.clip(np.asarray(confidence, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def bernoulli_nll(logits, labels):
    """Mean negative log likelihood of the labels under the logits."""
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def scaled_confidence(confidence, temperature):
    return expit(confidence_logit(confidence) / temperature)


def temperature_nll(temperature, logits, labels):
    return bernoulli_nll(logits / temperature, labels)


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
    return bernoulli_nll(weights[0] * logits + weights[1], labels)


def platt_gradient(weights, logits, labels):
    """Exact gradient of platt_nll, so the fit matches to machine precision anywhere."""
    residual = expit(weights[0] * logits + weights[1]) - labels
    return np.array([float(np.mean(residual * logits)), float(np.mean(residual))])


def platt_confidence(confidence, slope, intercept):
    return expit(slope * confidence_logit(confidence) + intercept)


def fit_platt(confidence, labels):
    """Slope and intercept by Bernoulli negative log likelihood. Convex in both, so
    the start point cannot move the answer. Identity when the labels carry no signal."""
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0 or labels.min() == labels.max():
        return 1.0, 0.0
    logits = confidence_logit(confidence)
    result = minimize(platt_nll, PLATT_START, jac=platt_gradient,
                      args=(logits, labels), method="BFGS")
    return float(result.x[0]), float(result.x[1])