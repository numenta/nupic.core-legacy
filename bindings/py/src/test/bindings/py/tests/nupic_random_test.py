#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""NuPIC random module tests."""

import cPickle as pickle
import unittest
import pytest
import numpy


from nupic.bindings.math import Random



class TestNupicRandom(unittest.TestCase):

  @pytest.fixture(autouse=True)
  def initdir(self, tmpdir):
    tmpdir.chdir() # change to the pytest-provided temporary directory
	
  def testSerialization(self):
    """Test serialization of NuPIC randomness."""
	
    path = "RandomSerialization.stream"

    # Simple test: make sure that dumping / loading works...
    r = Random(99)

    r.saveToFile(path)

    test1 = [r.getUInt32() for _ in xrange(10)]
    r = Random(1);
    r.loadFromFile(path)
    self.assertEqual(r.getSeed(), 99)
    test2 = [r.getUInt32() for _ in xrange(10)]

    self.assertEqual(test1, test2,
				"Simple NuPIC random serialization check failed.")

    # A little tricker: dump / load _after_ some numbers have been generated
    # (in the first test).  Things should still work...
    # ...the idea of this test is to make sure that the pickle code isn't just
    # saving the initial seed...
    r.saveToFile(path)

    test3 = [r.getUInt32() for _ in xrange(10)]
    r = Random();
    r.loadFromFile(path)
    self.assertEqual(r.getSeed(), 99)
    test4 = [r.getUInt32() for _ in xrange(10)]

    self.assertEqual(
        test3, test4,
        "NuPIC random serialization check didn't work for saving later state.")

    self.assertNotEqual(
        test1, test3,
        "NuPIC random serialization test gave the same result twice?!?")


  def testNupicRandomPickling(self):
    """Test pickling / unpickling of NuPIC randomness."""

    # Simple test: make sure that dumping / loading works...
    r = Random(42)
    pickledR = pickle.dumps(r)

    test1 = [r.getUInt32() for _ in xrange(10)]
    r = pickle.loads(pickledR)
    test2 = [r.getUInt32() for _ in xrange(10)]

    self.assertEqual(test1, test2,
                     "Simple NuPIC random pickle/unpickle failed.")

    # A little tricker: dump / load _after_ some numbers have been generated
    # (in the first test).  Things should still work...
    # ...the idea of this test is to make sure that the pickle code isn't just
    # saving the initial seed...
    pickledR = pickle.dumps(r)

    test3 = [r.getUInt32() for _ in xrange(10)]
    r = pickle.loads(pickledR)
    test4 = [r.getUInt32() for _ in xrange(10)]

    self.assertEqual(
        test3, test4,
        "NuPIC random pickle/unpickle didn't work for saving later state.")

    self.assertNotEqual(test1, test3,
                        "NuPIC random gave the same result twice?!?")


  def testSample(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([2], dtype="uint32")

    r.sample(population, choices)

    self.assertEqual(choices[0], 1)
    self.assertEqual(choices[1], 3)


  def testSampleNone(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([0], dtype="uint32")

    # Just make sure there is no exception thrown.
    r.sample(population, choices)

    self.assertEqual(choices.size, 0)


  def testSampleAll(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([4], dtype="uint32")

    r.sample(population, choices)

    self.assertEqual(choices[0], 1)
    self.assertEqual(choices[1], 2)
    self.assertEqual(choices[2], 3)
    self.assertEqual(choices[3], 4)


  def testSampleWrongDimensionsPopulation(self):
    """Check that passing a multi-dimensional array throws a ValueError."""
    r = Random(42)
    population = numpy.array([[1, 2], [3, 4]], dtype="uint32")
    choices = numpy.zeros([2], dtype="uint32")

    self.assertRaises(ValueError, r.sample, population, choices)


  def testSampleWrongDimensionsChoices(self):
    """Check that passing a multi-dimensional array throws a ValueError."""
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([2, 2], dtype="uint32")

    self.assertRaises(ValueError, r.sample, population, choices)


  def testSampleSequenceRaisesTypeError(self):
    """Check that passing lists throws a TypeError.

    This behavior may change if sample is extended to understand sequences.
    """
    r = Random(42)
    population = [1, 2, 3, 4]
    choices = [0, 0]

    self.assertRaises(TypeError, r.sample, population, choices)


  def testSampleBadDtype(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="int64")
    choices = numpy.zeros([2], dtype="int64")

    self.assertRaises(TypeError, r.sample, population, choices)


  def testSampleDifferentDtypes(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([2], dtype="uint64")

    self.assertRaises(ValueError, r.sample, population, choices)


  def testSamplePopulationTooSmall(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")
    choices = numpy.zeros([5], dtype="uint32")

    self.assertRaises(
        ValueError, r.sample, population, choices)


  def testShuffle(self):
    r = Random(42)
    arr = numpy.array([1, 2, 3, 4], dtype="uint32")

    r.shuffle(arr)

    self.assertEqual(arr[0], 1)
    self.assertEqual(arr[1], 4)
    self.assertEqual(arr[2], 3)
    self.assertEqual(arr[3], 2)


  def testShuffleEmpty(self):
    r = Random(42)
    arr = numpy.zeros([0], dtype="uint32")

    r.shuffle(arr)

    self.assertEqual(arr.size, 0)


  def testShuffleEmpty(self):
    r = Random(42)
    arr = numpy.zeros([2, 2], dtype="uint32")

    self.assertRaises(ValueError, r.shuffle, arr)


  def testShuffleBadDtype(self):
    r = Random(42)
    arr = numpy.array([1, 2, 3, 4], dtype="int64")

    self.assertRaises(ValueError, r.shuffle, arr)


  def testEquals(self):
    r1 = Random(42)
    v1 = r1.getReal64()
    r2 = Random(42)
    v2 = r2.getReal64()
    self.assertEquals(v1, v2)
    self.assertEquals(r1, r2)

if __name__ == "__main__":
  unittest.main()
