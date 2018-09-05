"""
XENON1T experiment.

See e.g., https://education.jlab.org/itselemental/iso054.html for the
natural isotopes of Xenon.
"""

import os.path

from .experiment import Experiment, Component


THIS = os.path.split(os.path.abspath(__file__))[0]
EFF_FILE = os.path.join(THIS, "./data/xenon.txt")


COMPONENTS = [Component(0.0191, 128, EFF_FILE),
              Component(0.264, 129, EFF_FILE),
              Component(0.0407, 130, EFF_FILE),
              Component(0.212, 131, EFF_FILE),
              Component(0.269, 132, EFF_FILE),
              Component(0.104, 134, EFF_FILE),
              Component(0.0886, 136, EFF_FILE)]

# This is the reference region of the inner volume

XENON1T = Experiment(0.475 * 900. * 278.8, 1.62, 2, COMPONENTS)

if __name__ == "__main__":
    for component in COMPONENTS:
        component.plot_efficiency()
        component.plot_energy_integral()

    print XENON1T.signal_upper_limit(0.9)
