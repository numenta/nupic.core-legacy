#!/usr/bin/env python
# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc.
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
# ----------------------------------------------------------------------

"""NuPIC random module tests."""

import pickle
import unittest
import pytest
import numpy
import sys


from htm.bindings.math import Random



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

    test1 = [r.getUInt32() for _ in range(10)]
    r = Random(1);
    r.loadFromFile(path)
    self.assertEqual(r.getSeed(), 99)
    test2 = [r.getUInt32() for _ in range(10)]

    self.assertEqual(test1, test2,
				"Simple NuPIC random serialization check failed.")

    # A little tricker: dump / load _after_ some numbers have been generated
    # (in the first test).  Things should still work...
    # ...the idea of this test is to make sure that the pickle code isn't just
    # saving the initial seed...
    r.saveToFile(path)

    test3 = [r.getUInt32() for _ in range(10)]
    r = Random();
    r.loadFromFile(path)
    self.assertEqual(r.getSeed(), 99)
    test4 = [r.getUInt32() for _ in range(10)]

    self.assertEqual(
        test3, test4,
        "NuPIC random serialization check didn't work for saving later state.")

    self.assertNotEqual(
        test1, test3,
        "NuPIC random serialization test gave the same result twice?!?")

  @pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails for python2 with segmentation fault")
  def testNupicRandomPickling(self):
    """Test pickling / unpickling of NuPIC randomness."""

    # Simple test: make sure that dumping / loading works...
    r = Random(42)
    pickledR = pickle.dumps(r)

    test1 = [r.getUInt32() for _ in range(10)]
    r = pickle.loads(pickledR)
    test2 = [r.getUInt32() for _ in range(10)]

    self.assertEqual(test1, test2,
                     "Simple NuPIC random pickle/unpickle failed.")

    # A little tricker: dump / load _after_ some numbers have been generated
    # (in the first test).  Things should still work...
    # ...the idea of this test is to make sure that the pickle code isn't just
    # saving the initial seed...
    pickledR = pickle.dumps(r)

    test3 = [r.getUInt32() for _ in range(10)]
    r = pickle.loads(pickledR)
    test4 = [r.getUInt32() for _ in range(10)]

    self.assertEqual(
        test3, test4,
        "NuPIC random pickle/unpickle didn't work for saving later state.")

    self.assertNotEqual(test1, test3,
                        "NuPIC random gave the same result twice.")
    self.assertNotEqual(test2, test4,
                        "NuPIC random gave the same result twice.")

  def testSample(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")

    choices = r.sample(population, 2)

    self.assertEqual(choices[0], 2)
    self.assertEqual(choices[1], 1)


  def testSampleNone(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")

    # Just make sure there is no exception thrown.
    choices = r.sample(population, 0)

    self.assertEqual(len(choices), 0)


  def testSampleAll(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")

    choices = r.sample(population, 4)

    self.assertEqual(choices[0], 2)
    self.assertEqual(choices[1], 1)
    self.assertEqual(choices[2], 4)
    self.assertEqual(choices[3], 3)


  def testSampleWrongDimensionsPopulation(self):
    """Check that passing a multi-dimensional array throws a ValueError."""
    r = Random(42)
    population = numpy.array([[1, 2], [3, 4]], dtype="uint32")

    self.assertRaises(ValueError, r.sample, population, 2)


  def testSampleSequenceRaisesTypeError(self):
    """Check that passing lists throws a TypeError.

    This behavior may change if sample is extended to understand sequences.
    """
    r = Random(42)
    population = [1, 2, 3, 4]

    self.assertRaises(TypeError, r.sample, population, 2)


  def testSampleBadDtype(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="int64")

    # throws std::invalid_argument("Invalid numpy array precision used.");
    # in py_utils.hpp
    # so thats why it is ValueError and not TypeError
    self.assertRaises(ValueError, r.sample, population, 2)


  def testSamplePopulationTooSmall(self):
    r = Random(42)
    population = numpy.array([1, 2, 3, 4], dtype="uint32")

    #RuntimeError and not ValueError because it goes to Cpp code and there is
    #just NTA_CHECK that raises runtime_error
    self.assertRaises(
        RuntimeError, r.sample, population, 999)


  def testShuffle(self):
    r = Random(42)
    arr = numpy.array([1, 2, 3, 4], dtype="uint32")

    r.shuffle(arr)

    self.assertEqual(arr[0], 2)
    self.assertEqual(arr[1], 1)
    self.assertEqual(arr[2], 4)
    self.assertEqual(arr[3], 3)


  def testShuffleEmpty(self):
    r = Random(42)
    arr = numpy.zeros([0], dtype="uint32")

    r.shuffle(arr)

    self.assertEqual(arr.size, 0)


  def testShuffleEmpty2(self):
    r = Random(42)
    arr = numpy.zeros([2, 2], dtype="uint32")#2x2 array dimension

    self.assertRaises(ValueError, r.shuffle, arr)


  def testShuffleBadDtype(self):
    r = Random(42)
    arr = numpy.array([1, 2, 3, 4], dtype="int64")

    self.assertRaises(ValueError, r.shuffle, arr)


  def testEquals(self):
    r1 = Random(42)
    v1 = r1.getReal64()
    i1 = r1.getUInt32()
    
    r2 = Random(42)
    v2 = r2.getReal64()
    i2 = r2.getUInt32()
    
    r3 = Random(66)
    
    self.assertEqual(v1, v2)
    self.assertEqual(i1, i2)
    self.assertEqual(r1,r2)
    self.assertNotEqual(r1,r3)
    

  def testPlatformSame(self): 
    r = Random(42)
    [r.getUInt32() for _ in range(80085)]
    v = r.getUInt32()
    self.assertEqual(v, 1651991554)

if __name__ == "__main__":
  unittest.main()
