"""
Boost from Earth frame to galactic one and angle average.
"""


import numpy as np
from numpy import trapz


BINS = 100
COS_THETA = np.linspace(-1., 1., BINS)


def v_earth_frame(v_galactic_frame, cos_theta, shift_to_earth):
    """
    @returns Velocity in earth frame
    """
    v_2 = (v_galactic_frame**2 + shift_to_earth**2
           + 2. * v_galactic_frame * cos_theta * shift_to_earth)
    v_earth_frame_ = v_2**0.5
    return v_earth_frame_

def angle_average(galactic_frame_func):
    """
    @returns Angle-averaged function
    """
    @np.vectorize
    def angle_averaged(v_galactic_frame):
        """
        @returns Angle-averaged evaluation of function
        """
        v = v_galactic_frame * np.ones_like(COS_THETA)
        integrand = galactic_frame_func(v, COS_THETA)
        return 0.5 * trapz(integrand, COS_THETA)

    return angle_averaged
