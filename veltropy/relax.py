"""
Marginalize uncertainty in velocity profile.
"""

from collections import Counter
import numpy as np
from scipy.special import gamma
from scipy.special import beta as beta_function
from scipy.misc import factorial, comb
from numpy import exp, prod, log

from .partition import partition
from .memoize import LookUpTable
from .poisson import Poisson


class Relax(Poisson):
    """
    Relax assumptions about a default distribution, e.g. a Maxwellian, using
    a multinomial prior.
    """
    def __init__(self, beta, signal, velocity_dist=None, isotropic=True):
        """
        @param beta Number of trials in multiomial distribution
        @param signal Events function
        """
        self.beta = beta if beta > 1. else 1.
        self.flatten = beta if beta < 1. else None
        super(Relax, self).__init__(signal, velocity_dist, isotropic)

    @LookUpTable
    def _shift(self, sigma):
        """
        Convenient to shift quantities in log-space to prevent nans etc
        """
        events = self.events * sigma / self.sigma
        power = -events / self.beta
        return np.nanmax(power[self.default.nonzero()])

    @LookUpTable
    def _modified(self, sigma):
        """
        @returns An expectation appearing in multinomial formula
        """
        events = self.events * sigma / self.sigma
        power = -events / self.beta - self._shift(sigma)
        modified = self.default * np.nan_to_num(exp(power))
        return modified

    @LookUpTable
    def _lognorm(self, sigma):
        """
        @returns An expectation appearing in multinomial formula
        """
        norm = self.dtrapz(self._modified(sigma))
        lognorm = log(norm)
        jensen = -self.signal(sigma) / self.beta - self._shift(sigma)
        if not lognorm >= jensen:
            lognorm = jensen
        return lognorm

    @LookUpTable
    def _mean(self, power, sigma):
        """
        @returns An expectation appearing in multinomial formula
        """
        if power == 0:
            return exp(self._lognorm(sigma))

        events = self.events * sigma / self.sigma
        integrand = self._modified(sigma) * (events + self.background)**power
        mean = self.dtrapz(integrand)
        return mean

    @LookUpTable
    def _coeff(self, powers):
        """
        @returns Coefficient of a term in multinomial formula
        """
        x = len([p for p in powers if p != 0])
        y = self.beta - x + 1
        falling_factorial = gamma(x) / beta_function(x, y)

        count = Counter(powers).values()
        F = prod([factorial(c) for c in count])

        q = sum(powers)
        choose = 1.
        for p in powers:
            choose *= comb(q, p)
            q -= p

        coeff = falling_factorial * choose / (F * self.beta**self.observed)
        return coeff

    def _prod_mean(self, powers, sigma):
        """
        @returns Product of means appearing in multinomial formula
        """
        return prod([self._mean(power, sigma) for power in powers])

    def _term(self, powers, sigma):
        """
        @returns Term in multinomial formula
        """
        powers = tuple(list(powers) + [0] * (self.observed - len(powers)))
        return self._coeff(powers) * self._prod_mean(powers, sigma)

    @LookUpTable
    def loglike(self, sigma):
        """
        @returns Log of averaged likelihood
        """
        log_factor = ((self.beta - self.observed) * self._lognorm(sigma)
                      - self.background + self.beta * self._shift(sigma))
        assert not np.isnan(log_factor)
        sum_ = sum([self._term(powers, sigma)
                    for powers in partition(self.observed)])
        assert not np.isnan(sum_)
        loglike = log_factor + log(sum_) - log(factorial(self.observed))
        assert loglike <= 0.
        return loglike

    def set_velocity_dist(self, velocity_dist):
        """
        Set the velocity distribution, taking special care of analytic
        continuation for beta < 1.
        """
        super(Relax, self).set_velocity_dist(velocity_dist)

        if self.flatten is not None:
            self.default = self.default**self.flatten
            norm = self.dtrapz(self.default)
            self.default /= norm

if __name__ == "__main__":

    from .events import EventsAtVelocity

    SIGMA = 2E-46
    MASS = 50.
    BETA = 1.

    SIGNAL = EventsAtVelocity(MASS)

    RELAX = Relax(BETA, SIGNAL)
    print RELAX(SIGMA)
