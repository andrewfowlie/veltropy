"""
Maxwell-Boltzmann distribution
"""

from abc import ABCMeta, abstractmethod
from numpy import pi, exp
from scipy.special import erf
from scipy.integrate import quad

import numpy as np
import matplotlib.pyplot as plt


class VelocityFunction(object):
    """
    Abstract class for function of velocity. Velocity typically in dimension
    of km/s.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, velocity, cos_theta=None):
        pass

    @property
    def min_v(self):
        """
        @returns Minimum velocity
        """
        return 0.

    @property
    def max_v(self):
        """
        @returns Maximum velocity
        """
        return 1E3

    def plot(self):
        """
        Plot the function of velocity
        """
        velocity = np.linspace(self.min_v, self.max_v, 100)
        plt.plot(velocity, self(velocity), lw=3, c="RoyalBlue")
        plt.xlabel("$v$")
        plt.ylabel("$f(v)$")
        plt.ylim(0., None)
        plt.show()

    def __mul__(self, other):
        """
        @returns Integral of product of two functions of velocity
        """
        def integrand(v):
            """
            @returns Product of two functions of velocity
            """
            return self(v) * other(v)

        lower = min(self.min_v, other.min_v)
        upper = max(self.max_v, other.max_v)
        return quad(integrand, lower, upper)[0]

class MB(VelocityFunction):
    """
    Truncated Maxwell-Boltzmann distribution. This is in the galactic frame.
    """
    def __init__(self, v_mode=235., v_escape=550.):
        self.v_mode = v_mode
        self.v_escape = v_escape
        self._norm = (0.25 * v_mode**2 *
                      (pi**0.5 * v_mode * erf(v_escape / v_mode)
                       - 2. * v_escape * exp(-(v_escape / v_mode)**2)))

    @property
    def max_v(self):
        return self.v_escape

    def __call__(self, velocity, cos_theta=None):
        mask = velocity <= self.max_v
        fail = np.zeros_like(velocity)
        power = -(velocity / self.v_mode)**2
        angular = 2. if cos_theta is not None else 1.
        pdf = velocity**2 * exp(power, where=mask, out=fail)
        pdf /= angular * self._norm
        return pdf

class Uniform(VelocityFunction):
    """
    Uniform distribution. This is in the galactic frame.
    """
    def __init__(self, v_escape=550.):
        self.v_escape = v_escape

    @property
    def max_v(self):
        return self.v_escape

    def __call__(self, velocity, cos_theta=None):
        angular = 2. if cos_theta is not None else 1.
        pdf = np.ones_like(velocity)
        pdf[velocity > self.v_escape] = 0.
        pdf /= angular * self.v_escape
        return pdf

if __name__ == "__main__":
    D = MB()
    D.plot()
