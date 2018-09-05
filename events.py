"""
Construct event function from a DD experiment.
"""

import numpy as np

from .velocity import VelocityFunction
from .angle_average import angle_average, v_earth_frame
from .xenon import XENON1T
from .memoize import LookUpTable


DEFAULT_SHIFT_TO_EARTH = 235.  # km/s
SPEED_OF_LIGHT = 3.E5  # km/s
KM_OVER_CM = 1E5
M_PROTON = 938.27231E-3  # GeV


class EventsAtVelocity(VelocityFunction):
    """
    Expected number of signal events as a function of velocity in the galactic
    frame.
    """
    SIGMA = 1E-46  # cm^2
    RHO = 0.3  # GeV/cm^2

    def __init__(self, mass, experiment=None, shift_to_earth=None):
        """
        @param mass Mass of DM in GeV
        @param shift_to_earth Shift to earth frame in km/s
        """
        self.mass = mass
        self.experiment = experiment or XENON1T
        self.shift_to_earth = shift_to_earth or DEFAULT_SHIFT_TO_EARTH
        # Velocity and component independent factor in unit GeV * km/s
        self.factor = (0.5 * KM_OVER_CM * SPEED_OF_LIGHT**2 *
                       self.experiment.exposure * self.RHO * self.SIGMA
                       / (self.mass * self._reduced_mass(M_PROTON)**2))

    def _reduced_mass(self, mass):
        """
        @returns Reduced mass of DM particle and other in GeV
        """
        return mass * self.mass / (mass + self.mass)

    def _max_energy(self, velocity, mass):
        """
        @param mass Mass of component of detector
        @returns Maximum energy from kinematics in GeV
        """
        return 2. * (self._reduced_mass(mass) * velocity
                     / SPEED_OF_LIGHT)**2 / mass

    @LookUpTable
    def events_earth_frame(self, velocity):
        """
        @returns Expected number of events as function of velocity in earth
        frame
        """
        max_energies = np.array([self._max_energy(velocity, c.mass)
                                 for c in self.experiment.components])
        sum_ = sum(c.contribution(e)
                   for e, c in zip(max_energies, self.experiment.components))
        return sum_ * self.factor / velocity

    @LookUpTable
    def events_galactic_frame(self, velocity, cos_theta):
        """
        @returns Expected number of events as function of velocity in galactic
        frame
        """
        v_earth_frame_ = v_earth_frame(velocity, cos_theta, self.shift_to_earth)
        return self.events_earth_frame(v_earth_frame_)

    @LookUpTable
    def averaged_events_galactic_frame(self, velocity):
        """
        @returns Expected number of events as function of velocity in galactic
        frame, angle-averaged
        """
        angle_averaged = angle_average(self.events_galactic_frame)
        return angle_averaged(velocity)

    def __call__(self, velocity, cos_theta=None):
        if cos_theta is None:
            return self.averaged_events_galactic_frame(velocity)
        return self.events_galactic_frame(velocity, cos_theta)

if __name__ == "__main__":

    MASS = 60.
    E = EventsAtVelocity(MASS)
    E.plot()
