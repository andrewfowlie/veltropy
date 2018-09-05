"""
Marginalize the shape parameters of the Maxwell-Boltzmann.
"""

import nestle
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .poisson import Poisson
from .velocity import MB
from .memoize import LookUpTable


def bound(pair, x):
    """
    @returns Whether parameter in bound
    """
    return pair[0] - 3. * pair[1] <= x <= pair[0] + 3. * pair[1]

def gauss(pair, x):
    """
    @returns Chi-squared from a nuisance parameter
    """
    return ((pair[0] - x) / pair[1])**2

class Shape(Poisson):
    """
    Average uncertainty in shape parameters.
    """
    V_ESCAPE = (550., 35.)
    V_MODE = (235., 20.)

    def __init__(self, treatment, mode=0):
        self.treatment = treatment
        self.mode = mode
        self.background = self.treatment.background
        self.observed = self.treatment.observed
        self.sigma = self.treatment.sigma

    @LookUpTable
    def nested(self, sigma):
        """
        @returns Nested sampling integration of shape parameters
        """
        def loglike(shape):
            """
            @returns Log-likelihood
            """
            self.set_velocity_dist(MB(*shape))
            return self.treatment.loglike(sigma)

        def prior(x):
            """
            @returns Shape parameters sampled from Gaussians
            """
            return (norm.ppf(x[0], *self.V_MODE), norm.ppf(x[1], *self.V_ESCAPE))

        result = nestle.sample(loglike, prior, 2, maxcall=1000, dlogz=0.5, npoints=25)
        return result

    def average(self, sigma):
        """
        @returns Log-likelihood averaged over shape parameters
        """
        result = self.nested(sigma)
        return result.logz

    def profile(self, sigma, gaussian=True):
        """
        @returns Log-likelihood profiled over shape parameters
        """
        def bounds(shape):
            """
            @returns Whether parameters in bounds
            """
            bounds = bound(self.V_MODE, shape[0]) and bound(self.V_ESCAPE, shape[1])
            return 0. if bounds else np.inf

        def gaussians(shape):
            """
            @returns Chi-squared from nuisance parameters
            """
            gaussians = gauss(self.V_MODE, shape[0]) + gauss(self.V_ESCAPE, shape[1])
            return gaussians

        def objective(shape):
            """
            @returns Chi-squared for particular shape parameters
            """
            nuisance = gaussians(shape) if gaussian else bounds(shape)
            if np.isinf(nuisance):
                return np.inf
            self.set_velocity_dist(MB(*shape))
            return self.treatment.chi_squared(sigma) + nuisance

        guess = (self.V_MODE[0], self.V_ESCAPE[0])
        sol = minimize(objective, guess, tol=1E-6, method="Powell")
        loglike = -0.5 * sol.fun
        return loglike

    def loglike(self, sigma):
        """
        @returns Log-likelihood either averaged or profiled
        """
        if self.mode == 0:
            return self.average(sigma)
        elif self.mode == 1:
            return self.profile(sigma, gaussian=False)
        elif self.mode == 2:
            return self.profile(sigma, gaussian=True)
        else:
            raise RuntimeError()

    def signal(self, sigma):
        """
        @returns Signal averaged over shape parameters
        """
        def signal(shape):
            """
            @returns Log-likelihood
            """
            self.set_velocity_dist(MB(*shape))
            return self.treatment.signal(sigma)

        result = self.nested(sigma)
        return np.sum([signal(shape) * weight
                       for shape, weight in zip(result.samples, result.weights)])

    def set_velocity_dist(self, velocity_dist):
        """
        Set the velocity distribution
        """
        self.treatment.set_velocity_dist(velocity_dist)

if __name__ == "__main__":

    from .events import EventsAtVelocity

    SIGMA = 1E-46
    MASS = 30.
    SIGNAL = EventsAtVelocity(MASS)

    POISSON = Poisson(SIGNAL)
    print POISSON.chi_squared_limit()

    SHAPE = Shape(POISSON, mode=1)
    print SHAPE.chi_squared_limit()

    SHAPE = Shape(POISSON, mode=2)
    print SHAPE.chi_squared_limit()
