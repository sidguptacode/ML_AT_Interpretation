"""The NGBoost Gamma distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import gamma as dist
import ngboost
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

eps = 1e-10

class GammaLogScore(LogScore):
    def score(self, Y):
        _score = -self.dist.logpdf(Y + eps)
        return _score

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dG/d(log(alpha)), second col is dG/d(log(beta))
        # All terms here are finite (checked)
        D[:, 0] = self.beta * (Y + eps) - self.alpha
        # All terms here are finite (checked)
        first_term = np.log(self.alpha)
        second_term = np.log(self.beta)
        third_term = np.log(Y + eps)
        D[:, 1] = self.alpha * (first_term - second_term - third_term) - 0.5
        return D


class Gamma(RegressionDistn):
    """
    Implements the gamma distribution for NGBoost.

    The gamma distribution has two parameters: alpha and beta. See scipy.stats.gamma for details.
    This distribution has both LogScore and CRPScore implemented for it
    and both work with right-censored data
    """

    n_params = 2
    scores = [GammaLogScore]

    def __init__(self, params):  # pylint: disable=super-init-not-called
        self._params = params
        # In NGBoost, all parameters must be represented internally in ℝ, so we need to reparametrize (alpha,beta) to, for instance, (log(alpha),log(beta)). 
        # The latter are the parameters we need to work with when we initialize a Gamma object and when implement the score.
        self.alpha, self.beta = np.exp(params[0]), np.exp(params[1])
        # Scale = 1/beta, as defined in scipy
        self.dist = dist(a = self.alpha, scale=1/self.beta)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    # should implement a `sample()` method
    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}

    def fit(Y):
        fit_alpha, fit_loc, fit_beta = sp.stats.gamma.fit(Y) # NOTE: maybe fit_beta is actually scale? so beta is 1/fit_beta???
        return np.array([np.log(fit_alpha), np.log(1/fit_beta)])

    # This is implemented indirectly with the __getattr_ method
    # def mean(self):
    #     return self.alpha / self.beta