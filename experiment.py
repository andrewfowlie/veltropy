"""
Classes for an experiment e.g., XENON1T.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from scipy.integrate import cumtrapz
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from memoized_property import memoized_property

from .memoize import LookUpTable
from .form_factor import helm_form_factor


KG_DAYS_TO_GEV_SECONDS = 86400. / 1.783E-27
ATOMIC_MASS_TO_GEV = 931.494E-3
KEV_TO_GEV = 1E-6


@LookUpTable
def signal_upper_limit(background, observed, level):
    """
    @param level Confidence level
    @returns Upper limit on number of signal events
    """
    def objective(signal):
        """
        @returns Objective for which we want to find zero crossing
        """
        cdf = poisson.cdf(observed, signal + background)
        return cdf - (1. - level)
    return bisect(objective, 0, 1000., xtol=1E-3, rtol=1E-3)

class Component(object):
    """
    Class for component of a detector.
    """
    def __init__(self, fraction, A, efficiency_file):
        self.fraction = fraction
        self.A = A
        self.efficiency_file = efficiency_file

    @memoized_property
    def mass(self):
        """
        Mass of nucleus in GeV.
        """
        return self.A * ATOMIC_MASS_TO_GEV

    @memoized_property
    def efficiency(self):
        """
        Detector efficiency (dimensionless) as a function of energy in GeV.
        """
        energy, efficiency = np.loadtxt(self.efficiency_file, unpack=True)
        return interp1d(energy * KEV_TO_GEV, efficiency,
                        bounds_error=False, fill_value=0.)

    @memoized_property
    def max_energy(self):
        """
        Maximum energy (GeV).
        """
        energy, _ = np.loadtxt(self.efficiency_file, unpack=True)
        return max(energy) * KEV_TO_GEV

    @memoized_property
    def tabulate_energy(self):
        """
        Energies for which to tabulate energy integral (GeV).
        """
        return np.logspace(-20., np.log10(self.max_energy), 1000)

    @memoized_property
    def energy_integral(self):
        """
        Tablulate energy integral per experiment. Unit of GeV.
        """
        def integrand(energy):
            """
            @returns Integrand
            """
            efficiency = self.efficiency(energy)
            q = (2. * self.mass * energy)**0.5
            form_factor = helm_form_factor(q, self.A)
            return form_factor**2 * efficiency

        integrand_ = integrand(self.tabulate_energy)
        cumulative = cumtrapz(integrand_, self.tabulate_energy)
        center = 0.5 * (self.tabulate_energy[:-1] + self.tabulate_energy[1:])
        return interp1d(center, cumulative,
                        bounds_error=False, fill_value=(0., cumulative[-1]))

    def contribution(self, max_energy):
        """
        Contribution from detector component, in unit 1 / GeV.
        """
        return self.A**2 * self.fraction * self.energy_integral(max_energy)

    def plot_efficiency(self):
        """
        Plot efficiency as a function of energy.
        """
        energy = np.linspace(0., self.max_energy, 1000)
        efficiency = self.efficiency(energy)
        plt.plot(energy, efficiency, lw=3, c="RoyalBlue")
        plt.xlabel("$E$")
        plt.ylabel(r"$\phi(E)$")
        plt.show()

    def plot_energy_integral(self):
        """
        Plot energy integral as a function of energy.
        """
        integral = self.energy_integral(self.tabulate_energy)
        plt.plot(self.tabulate_energy, integral, lw=3, c="RoyalBlue")
        plt.xlabel("$E$")
        plt.ylabel(r"$F(E)$")
        plt.show()

class Experiment(object):
    """
    Direct detection experiment.
    """
    def __init__(self, exposure, background, observed, components):
        """
        @param expsure Expsoure in kg * days
        """
        self.exposure = exposure * KG_DAYS_TO_GEV_SECONDS
        self.background = background
        self.observed = observed
        self.components = components

        assert np.isclose(sum(c.fraction for c in components), 1., 0.01)

    def chi_squared(self, signal):
        """
        @param signal Number of signal events
        @returns Chi-squared for a number of signal events
        """
        return -2. * poisson.logpmf(self.observed, signal + self.background)

    @LookUpTable
    def signal_upper_limit(self, level):
        """
        @param level Confidence level
        @returns Upper limit on number of signal events
        """
        return signal_upper_limit(self.background, self.observed, level)
