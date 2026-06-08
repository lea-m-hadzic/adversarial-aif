"""Belief updates over the other agent's intent.

Hidden variable theta in {Y, N}: Yielder or Non-yielder. Fixed for an episode.
Belief p_t = P(theta = Y | observations up to t).

Observation per step:
    o_t = (a_t, s_t)
        a_t : observed acceleration (m/s^2)
        s_t : observed yielding signal (0 or 1)

Likelihood factorizes (conditional independence given theta):
    P(o_t | theta) = P(a_t | theta) * P(s_t | theta)

Acceleration likelihood: Gaussian.
    a_t | Y ~ N(mu_yield,   sigma^2)
    a_t | N ~ N(mu_noyield, sigma^2)

Signal likelihood: Bernoulli.
    P(s_t = 1 | Y) = q_yield      (close to 1)
    P(s_t = 1 | N) = q_noyield    (close to 0)

A "naively trusting" agent uses q_noyield near 0, so seeing signal=1 strongly
shifts the belief toward Yielder; this is the agent that gets fooled by
deceptive adversaries.
"""

from dataclasses import dataclass
import math


@dataclass
class LikelihoodModel:
    """Likelihood parameters for both acceleration and signal observations."""
    mu_yield: float = -2.0
    mu_noyield: float = 0.0
    sigma: float = 1.0
    q_yield: float = 0.95         # P(signal=1 | Y)
    q_noyield: float = 0.05       # P(signal=1 | N) — small => trusting

    def accel_log_likelihood_ratio(self, observed_accel: float) -> float:
        """log P(a | Y) - log P(a | N), Gaussian likelihood."""
        return (
            (self.mu_yield - self.mu_noyield)
            * (observed_accel - 0.5 * (self.mu_yield + self.mu_noyield))
            / (self.sigma ** 2)
        )

    def signal_log_likelihood_ratio(self, observed_signal: int) -> float:
        """log P(s | Y) - log P(s | N), Bernoulli likelihood."""
        if observed_signal == 1:
            return math.log(self.q_yield) - math.log(self.q_noyield)
        return math.log(1.0 - self.q_yield) - math.log(1.0 - self.q_noyield)


def logit(p: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def update(
    prior: float,
    observed_accel: float,
    observed_signal: int,
    model: LikelihoodModel,
) -> float:
    """One Bayesian update step using both acceleration and signal."""
    llr = (
        model.accel_log_likelihood_ratio(observed_accel)
        + model.signal_log_likelihood_ratio(observed_signal)
    )
    return sigmoid(logit(prior) + llr)
