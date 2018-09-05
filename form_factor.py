"""
Helm form-factor.
"""

import numpy as np
from numpy import exp, sin, cos, pi


GEV_TO_INVERSE_FM = 1.E-6 / 1.973E-7
S = 0.9
a = 0.52
C1 = 1.23
C2 = 0.6


def helm_form_factor(q, A=130):
    """
    @param q Momentum in GeV
    @param A Nucleon number
    @returns Form factor
    """
    q *= GEV_TO_INVERSE_FM

    def form_factor(q):
        """
        @returns Form factor without treatment of q -> 0.
        """
        c = C1 * A**(1. / 3.)  - C2
        r = (c**2 + 7. / 3. * pi**2 * a**2 - 5. * S**2)**0.5
        qr = q * r
        return 3. * exp(-0.5 * q**2 * S**2) * (sin(qr) - qr * cos(qr)) / qr**3

    return np.where(q > 0, form_factor(q), 1.)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from numpy import linspace

    Q = linspace(0., 1., 1000)
    FF = map(helm_form_factor, Q)
    plt.plot(Q, FF)
    plt.show()
