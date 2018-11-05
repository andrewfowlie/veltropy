"""
Canonical calculation of log-likelihood from a Poisson.
"""

from memoized_property import memoized_property

import numpy as np
from numpy import trapz, exp, log
from scipy.stats import poisson, chi2
from scipy.optimize import bisect, minimize
from warnings import warn

from .memoize import LookUpTable, erase_lookup_tables
from .velocity import MB
from .experiment import signal_upper_limit



def safe_bisect(crossing, lower, upper, **kwargs):
    """
    @returns Root found by bisection
    """
    for _ in range(10):
        try:
            return bisect(crossing, lower, upper, **kwargs)
        except ValueError:
            lower -= 5
            upper += 5
    raise RuntimeError("could not find root")


class Poisson(object):
    """
    Likelihood from default distribution with no non-parametric uncertainty.
    """
    BINS = 100
    VELOCITY = np.linspace(0., 750., BINS)
    COS_THETA = np.linspace(-1., 1., BINS)
    MESH = np.meshgrid(VELOCITY, COS_THETA)
    BOUNDS = log((1E-43, 1E-47))

    def __init__(self, signal, velocity_dist=None, isotropic=True):
        """
        """
        self.default = None
        self.events = None
        self.observed = signal.experiment.observed
        self.background = signal.experiment.background
        self.sigma = signal.SIGMA
        self.isotropic = isotropic
        velocity_dist = velocity_dist or MB()
        self.set_velocity_dist(velocity_dist)
        self.set_events(signal)

    @LookUpTable
    def loglike(self, sigma):
        """
        @returns Log-likelihood
        """
        signal = self.signal(sigma)
        expected = self.background + signal
        return poisson.logpmf(self.observed, expected)

    def dtrapz(self, integrand):
        """
        @returns Twice integrated by trapezium rule
        """
        return trapz(trapz(integrand, self.VELOCITY), self.COS_THETA)

    @LookUpTable
    def signal(self, sigma):
        """
        @returns Expected number of signal events
        """
        events = self.events * sigma / self.sigma
        integrand = self.default * events
        mean = self.dtrapz(integrand)
        return mean

    def erase_lookup_tables(self):
        """
        Clear any look-up tables.
        """
        erase_lookup_tables(self)

    def set_velocity_dist(self, velocity_dist):
        """
        Set the velocity distribution.
        """
        self.erase_lookup_tables()
        if self.isotropic:
            default = 0.5 * velocity_dist(self.VELOCITY)
            default = np.broadcast_to(default, (self.BINS, self.BINS))
        else:
            default = velocity_dist(*self.MESH)

        norm = self.dtrapz(default)
        self.default = default / norm

    def set_events(self, signal):
        """
        Set the events function.
        """
        self.erase_lookup_tables()
        self.events = signal(*self.MESH)

        if self.isotropic:
            averaged = 0.5 * trapz(self.events.T, self.COS_THETA)
            self.events = np.broadcast_to(averaged, (self.BINS, self.BINS))

    @LookUpTable
    def chi_squared(self, sigma):
        """
        @returns Chi-squared associated with averaged likelihood
        """
        return -2. * self.loglike(sigma)

    @memoized_property
    def minimum_chi_squared(self):
        """
        @returns Minimum chi-squared
        """
        return self.chi_squared(self.best_fit_sigma)

    @memoized_property
    def best_fit_sigma(self):
        """
        @returns Best-fit cross section
        """
        observed_signal = max(0., self.observed - self.background)
        guess = self.sigma * observed_signal / self.signal(self.sigma)

        if guess == 0.:
            warn("guess was zero; assume limit 0")
            return 0.
        elif np.isinf(guess) or np.isnan(guess):
            warn("guess was inf; assume limit inf")
            return np.inf

        def target(log_sigma):
            """
            @returns Function that we minimize
            """
            sigma = exp(log_sigma)[0]
            return self.chi_squared(sigma)

        try:
            log_sigma = minimize(target, log(guess), tol=1E-6, bounds=None, method="Powell").x
        except ValueError:
            warn("could not find limit; assume limit inf")
            return np.inf

        sigma = exp(log_sigma)
        return sigma

    @LookUpTable
    def poisson_limit(self, level=0.9):
        """
        @param level Confidence level for limit
        @returns Upper limit on scattering cross section
        """
        limit = signal_upper_limit(self.background, self.observed, level)
        try:
            return self.sigma * limit / self.signal(self.sigma)
        except ZeroDivisionError:
            warn("signal was zero; assume limit inf")
            return np.inf

    @LookUpTable
    def chi_squared_limit(self, level=0.9):
        """
        @warning We use a 1/2 chi-squared with 1 dof
        @param level Confidence level for limit
        @returns Upper limit on scattering cross section from chi-squared
        """
        if np.isinf(self.best_fit_sigma):
            warn("best-fit inf; assume limit inf")
            return np.inf

        critical = chi2.isf(2. * (1. - level), 1)  # 1/2 chi-squared with 1 dof
        goal = critical + self.minimum_chi_squared

        def crossing(log_sigma):
            """
            @returns Function that we desire zero-crossing
            """
            return self.chi_squared(exp(log_sigma)) - goal

        try:
            log_limit = safe_bisect(crossing, *self.BOUNDS, rtol=1E-6, xtol=1E-6)
        except RuntimeError:
            warn("could not find upper limit; assume inf")
            return np.inf

        return exp(log_limit)

    def __call__(self, sigma):
        return self.chi_squared(sigma)

if __name__ == "__main__":

    from .events import EventsAtVelocity

    SIGMA = 2E-46
    MASS = 100.

    SIGNAL = EventsAtVelocity(MASS)

    POISSON = Poisson(SIGNAL)
    print POISSON(SIGMA)
