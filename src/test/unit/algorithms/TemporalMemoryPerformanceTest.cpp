/* ---------------------------------------------------------------------
* Numenta Platform for Intelligent Computing (NuPIC)
* Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
* with Numenta, Inc., for a separate license for this software code, the
* following terms and conditions apply:
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License version 3 as
* published by the Free Software Foundation.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see http://www.gnu.org/licenses.
*
* http://numenta.org/licenses/
* ---------------------------------------------------------------------
*/

/** @file
* Implementation of unit tests for SpatialPooler
*/
/*
#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/test/Tester.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

//import time
//import unittest
//mport numpy

#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>

#include <nupic/algorithms/SpatialPooler.hpp>   // SP
#include <nupic/algorithms/Cells4.hpp>          // TP
#include <nupic/algorithms/TemporalMemory.hpp>  // TM

#define Cells4 TemporalPooler

using namespace std;
using namespace nupic::utils;
using namespace nupic::algorithms;

namespace nupic {

  // ==============================
  //  Tests
  // ==============================

class TemporalMemoryPerformanceTest : public Tester
{
  SpatialPooler   sp;
  TemporalPooler  tp;
  TemporalMemory  tm;
  PatternMachine  _patternMachine;
  SequenceMachine _sequenceMachine;

  void setUp()
  {  
    tm.initialize({2048}, 32, 15, 2048L, .5, .8, 15, 12, .1, .05);

    _patternMachine = PatternMachine(2048, 40, 100);
    _sequenceMachine = SequenceMachine(_patternMachine);


  def testSingleSequence(self):
    print "Test: Single sequence"
    sequence = self.sequenceMachine.generateFromNumbers(range(50))
    times = self._feedAll(sequence)

    self.assertTrue(times[0] < times[1])
    self.assertTrue(times[2] < times[1])
    self.assertTrue(times[2] < times[0])


  # ==============================
  # Helper functions
  # ==============================

  def _feedAll(self, sequence, learn=True, num=1):
    repeatedSequence = sequence * num
    times = []

    def tmComputeFn(pattern, instance):
      instance.compute(pattern, learn)

    def tpComputeFn(pattern, instance):
      array = self._patternToNumpyArray(pattern)
      instance.compute(array, enableLearn=learn, computeInfOutput=True)

    elapsed = self._feedOne(repeatedSequence, self.tm, tmComputeFn)
    times.append(elapsed)
    print "TM:\t{0}s".format(elapsed)

    elapsed = self._feedOne(repeatedSequence, self.tp, tpComputeFn)
    times.append(elapsed)
    print "TP:\t{0}s".format(elapsed)

    elapsed = self._feedOne(repeatedSequence, self.tp10x2, tpComputeFn)
    times.append(elapsed)
    print "TP10X2:\t{0}s".format(elapsed)

    return times


  @staticmethod
  def _feedOne(sequence, instance, computeFn):
    start = time.clock()

    for pattern in sequence:
      if pattern == None:
        instance.reset()
      else:
        computeFn(pattern, instance)

    elapsed = time.clock() - start

    return elapsed


  @staticmethod
  def _patternToNumpyArray(pattern):
    array = numpy.zeros(2048, dtype='int32')
    array[list(pattern)] = 1

    return array



# ==============================
# Main
# ==============================

if __name__ == "__main__":
  unittest.main()
*/