"""
Expectations of Dirichlet
=========================
"""

import numpy as np
from scipy.stats import dirichlet, poisson

from .poisson import Poisson


class Dirichlet(Poisson):
    """
    Average log-likelihood upon Dirichlet distribution.
    """
    def __init__(self, signal, velocity_dist=None, isotropic=True, n_bins=100, gosper=False):
        """
        @param signal Events function
        """
        self.BINS = n_bins
        self.VELOCITY = np.linspace(0., 550., self.BINS)  # Insure this is cut-off
        self.COS_THETA = np.linspace(-1., 1., self.BINS)
        self.MESH = np.meshgrid(self.VELOCITY, self.COS_THETA)

        super(Dirichlet, self).__init__(signal, velocity_dist, isotropic)

        k = self.BINS if isotropic else self.BINS**2
        self.alpha = np.ones(k) if gosper else 0.5 * np.ones(k)

        if isotropic:
            self.flat_events = self.events[0, :]
        else:
            self.flat_events = self.events.flatten()

    def draw(self, n=10000):
        """
        @param n Number of samples
        @returns Velocity distributions from Dirichlet
        """
        return dirichlet.rvs(self.alpha, n)

    def loglike(self, sigma):
        """
        @returns Log of average likelihood, averaged upon distributions
        drawn from Dirichlet.
        """
        d = self.draw()
        signal = np.dot(d, self.flat_events) * sigma / self.sigma
        expected = self.background + signal
        like_ = poisson.pmf(self.observed, expected)
        loglike_ = np.log(np.mean(like_))
        return loglike_


if __name__ == "__main__":

    from .events import EventsAtVelocity

    SIGMA = 2E-46
    MASS = 10.

    SIGNAL = EventsAtVelocity(MASS)
    DIRICHLET = Dirichlet(SIGNAL)

    print DIRICHLET(SIGMA)
    print DIRICHLET.chi_squared_limit()
