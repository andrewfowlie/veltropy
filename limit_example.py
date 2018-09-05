"""
Basic example of code. We plot the upper limit on the cross section
with an entropic prior.
"""

import numpy as np
import matplotlib.pyplot as plt

from veltropy import Relax, EventsAtVelocity, MB, XENON1T


# Hyperparameter of entropic prior

BETA = 10

# Whether to restrict to isotropic distributions

ISOTROPIC = True

# Default velocity distribution

VELOCITY = MB()

# XENON1T experiment

EXPERIMENT = XENON1T


mass = np.logspace(0., 4., 100)
limit = np.zeros_like(mass)

for i, m in enumerate(mass):

    # Events function for that mass for the XENON1T experiment
    signal = EventsAtVelocity(m, experiment=EXPERIMENT)

    # Find upper limit on cross section
    relax = Relax(BETA, signal, velocity_dist=VELOCITY, isotropic=ISOTROPIC)
    limit[i] = relax.chi_squared_limit()

plt.plot(mass, limit)

plt.yscale('log')
plt.xscale('log')
plt.xlim([1., 1E4])
plt.ylim([1E-47, 1E-43])
plt.xlabel("mass (GeV)")
plt.ylabel(r"cross section (cm2)")
plt.show()
