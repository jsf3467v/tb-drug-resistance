"""Temperature-scaling math for post-hoc probability calibration.

    p_scaled = sigmoid(logit(p) / T)

T > 1 softens over-confident scores toward 0.5; T < 1 sharpens them; T is
fit by minimizing Bernoulli NLL against observed correctness, which
leaves the ranking untouched and changes only the scaled probability.
"""

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-6
T_BOUNDS = (0.05, 10.0)


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