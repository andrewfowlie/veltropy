"""
Basic tests for code.
"""

import unittest
import numpy as np

from ..events import EventsAtVelocity
from ..relax import Relax
from ..poisson import Poisson
from ..shape import Shape
from ..velocity import MB


MASS = 100.
SIGMA = 1E-45
V = 300.


class Test(unittest.TestCase):
    """
    Test important classes
    """
    def check(self, expected, found, atol=1E-2, rtol=1E-2, **kwargs):
        """
        Float comparison
        """
        pass_ = np.isclose(expected, found, rtol, atol)
        if not pass_:
            message = "{} != {}. {}".format(expected, found, kwargs)
            raise self.failureException(message)

    def setUp(self):
        """
        Make events function etc.
        """
        self.signal = EventsAtVelocity(MASS)
        self.addTypeEqualityFunc(float, self.check)

    def test_maxwell(self):
        """
        Check events function
        """
        expected = 0.00310445859809
        found = float(MB()(V))
        self.assertEqual(expected, found)

    def test_events_at_v(self):
        """
        Check events function
        """
        expected = 4.05698985621
        found = float(self.signal(V))
        self.assertEqual(expected, found)

    def test_poisson(self):
        """
        Check ordinary likelihood
        """
        poisson = Poisson(self.signal)
        expected = 74.5188911974
        found = float(poisson(SIGMA))
        self.assertEqual(expected, found)

    def test_relaxed(self):
        """
        Check relaxed likelihood
        """
        beta = 1.
        relax = Relax(beta, self.signal)
        expected = 59.3135202478
        found = float(relax(SIGMA))
        self.assertEqual(expected, found)

    def test_anisotropic(self):
        """
        Check relaxed likelihood for anisotropic distributions
        """
        beta = 1.
        relax = Relax(beta, self.signal, isotropic=False)
        expected = 11.2528301232
        found = float(relax(SIGMA))
        self.assertEqual(expected, found)

    def test_shape(self):
        """
        Check relaxed likelihood + parametric uncertainty
        """
        beta = 1.
        shape = Shape(Relax(beta, self.signal))
        expected = 57.8961281542
        found = float(shape(SIGMA))
        self.assertEqual(expected, found)

if __name__ == '__main__':
    unittest.main()
