import unittest
import numpy as np

from bartpy.samplers.schedule import SampleSchedule


class TestSchedule(unittest.TestCase):

    def testSampleZi(self):
        n = 10000
        sample = np.zeros(n)
        for k in range(n):
            sample[k] = SampleSchedule.conditional_z(-0.87, 1)

        print("mean Z conditional to y == 1, min", np.mean(sample), min(sample))
        self.assertGreaterEqual(min(sample), 0)
