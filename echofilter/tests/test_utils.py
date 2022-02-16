"""
Tests for utils.py.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
