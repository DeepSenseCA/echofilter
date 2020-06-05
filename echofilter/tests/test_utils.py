"""
Tests for utils.py.
"""

import numpy as np

from .base_test import BaseTestCase, unittest
from .. import utils


class test_get_indicator_onoffsets(BaseTestCase):
    """
    Tests for the get_indicator_onoffsets function.
    """

    def test_empty(self):
        """Test with empty numpy array."""
        self.assertEqual(utils.get_indicator_onoffsets(np.array([])), ([], []))

    def test_zeros(self):
        """Test with all zeros input."""
        self.assertEqual(utils.get_indicator_onoffsets(np.zeros(6)), ([], []))

    def test_ones(self):
        """Test with all zeros input."""
        arr = np.ones(6)
        out = utils.get_indicator_onoffsets(arr)
        self.assertEqual(out, ([0], [5]))
        self.assertEqual(arr, arr[out[0][0] : out[1][0] + 1])

    def test_onoff(self):
        ons = [3, 8, 11]
        offs = [5, 9, 14]
        arr = np.zeros(20)
        for i, j in zip(ons, offs):
            arr[i : j + 1] = 1
        out = utils.get_indicator_onoffsets(arr)
        self.assertEqual(out, (ons, offs))

    def test_onoff_left(self):
        ons = [0, 8, 11]
        offs = [5, 9, 14]
        arr = np.zeros(20)
        for i, j in zip(ons, offs):
            arr[i : j + 1] = 1
        out = utils.get_indicator_onoffsets(arr)
        self.assertEqual(out, (ons, offs))

    def test_onoff_right(self):
        ons = [3, 8, 11]
        offs = [5, 9, 19]
        arr = np.zeros(20)
        for i, j in zip(ons, offs):
            arr[i : j + 1] = 1
        out = utils.get_indicator_onoffsets(arr)
        self.assertEqual(out, (ons, offs))
