"""
Provides a base test class for other test classes to inherit from.

Includes the numpy testing functions as methods.
"""

# Copyright (c) 2021 Scott Lowe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import contextlib
import os.path
import sys
from inspect import getsourcefile

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_less,
    assert_array_max_ulp,
    assert_equal,
    assert_raises,
    assert_string_equal,
    assert_warns,
)

# For Python < 2.7, unittest2 is a backport of unittest
if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest


class BaseTestCase(unittest.TestCase):
    """
    Superclass for test cases, including support for numpy.
    """

    # The attribute `test_directory` provides the path to the directory
    # containing the file `base_test.py`, which is useful to obtain
    # test resources - files which are needed to run tests.
    test_directory = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

    def __init__(self, *args, **kw):
        # First to the __init__ associated with parent class
        # NB: The new method is like so, but this only works on Python3
        # super(self).__init__(*args, **kw)
        # So we have to do this for Python2 to be supported
        super(BaseTestCase, self).__init__(*args, **kw)
        # Add a test to automatically use when comparing objects of
        # type numpy ndarray. This will be used for self.assertEqual().
        self.addTypeEqualityFunc(np.ndarray, self.assert_allclose)

    @contextlib.contextmanager
    def subTest(self, *args, **kwargs):
        # For backwards compatability with Python < 3.4
        # Gracefully degrades into no-op.
        if hasattr(super(BaseTestCase, self), "subTest"):
            yield super(BaseTestCase, self).subTest(*args, **kwargs)
        else:
            yield None

    # Add assertions provided by numpy to this class, so they will be
    # available as methods to all subclasses when we do our tests.
    def assert_almost_equal(self, *args, **kwargs):
        """
        Check if two items are not equal up to desired precision.
        """
        return assert_almost_equal(*args, **kwargs)

    def assert_approx_equal(self, *args, **kwargs):
        """
        Check if two items are not equal up to significant digits.
        """
        return assert_approx_equal(*args, **kwargs)

    def assert_array_almost_equal(self, *args, **kwargs):
        """
        Check if two objects are not equal up to desired precision.
        """
        return assert_array_almost_equal(*args, **kwargs)

    def assert_allclose(self, *args, **kwargs):
        """
        Check if two objects are equal up to desired tolerance.
        """
        # Handle msg argument, which is passed from assertEqual, established
        # with addTypeEqualityFunc in __init__
        kwargs.pop("msg", None)
        return assert_allclose(*args, **kwargs)

    def assert_array_almost_equal_nulp(self, *args, **kwargs):
        """
        Compare two arrays relatively to their spacing.
        """
        return assert_array_almost_equal_nulp(*args, **kwargs)

    def assert_array_max_ulp(self, *args, **kwargs):
        """
        Check that all items of arrays differ in at most N Units in the Last Place.
        """
        return assert_array_max_ulp(*args, **kwargs)

    def assert_array_equal(self, *args, **kwargs):
        """
        Check if two array_like objects are equal.
        """
        return assert_array_equal(*args, **kwargs)

    def assert_array_less(self, *args, **kwargs):
        """
        Check if two array_like objects are not ordered by less than.
        """
        return assert_array_less(*args, **kwargs)

    def assert_equal(self, *args, **kwargs):
        """
        Check if two objects are not equal.
        """
        return assert_equal(*args, **kwargs)

    def assert_raises(self, *args, **kwargs):
        """
        Check that an exception of class exception_class is thrown by callable.
        """
        return assert_raises(*args, **kwargs)

    def assert_warns(self, *args, **kwargs):
        """
        Check that the given callable throws the specified warning.
        """
        return assert_warns(*args, **kwargs)

    def assert_string_equal(self, *args, **kwargs):
        """
        Test if two strings are equal.
        """
        return assert_string_equal(*args, **kwargs)
