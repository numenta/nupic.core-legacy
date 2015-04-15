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

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/test/Tester.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

//import time
//import unittest
//import numpy

#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

#include <nupic/os/Timer.hpp>

using namespace std;
using namespace nupic::utils;
using namespace nupic::algorithms::spatial_pooler;
using namespace nupic::algorithms::Cells4;
using namespace nupic::algorithms::temporal_memory;

namespace nupic {

  class SpatialPoolerInstance : public Instance, public SpatialPooler
  {
  public:
    void compute(vector<UInt> pattern, bool learn) { };
    void compute(vector<UInt> pattern, bool learn, bool learn2) { };
  };

  class Cells4Instance : public Instance, public Cells4
  {
  public:
    void compute(vector<UInt> pattern, bool learn) { };
    void compute(vector<UInt> pattern, bool learn, bool learn2) { };
  };

  class TemporalMemoryInstance : public Instance, public TemporalMemory
  {
  public:
    void compute(vector<UInt> pattern, bool learn) { };
    void compute(vector<UInt> pattern, bool learn, bool learn2) { };
  };

  static void tmComputeFn(vector<UInt>& pattern, Instance& instance)
  {
    instance.compute(pattern, instance._learn);
  }

  static void tpComputeFn(vector<UInt>& pattern, Instance& instance)
  {
    instance.compute(pattern, instance._learn, true);
  }

  // ==============================
  //  Tests
  // ==============================

  class TemporalMemoryPerformanceTest : public Tester
  {
    SpatialPoolerInstance sp;
    Cells4Instance tp;
    TemporalMemoryInstance tm;
    PatternMachine  _patternMachine;
    SequenceMachine _sequenceMachine;
    bool _learn;

    void setUp()
    {
      tm.initialize({ 2048 }, 32, 15, 2048L, .5, .8, 15, 12, .1, .05);

      _patternMachine = PatternMachine();
      _patternMachine.initialize(2048, vector<UInt>{ 40 }, 100);
      _sequenceMachine = SequenceMachine(_patternMachine);
    }

    // Run all appropriate tests
    virtual void RunTests() override
    {
      testSingleSequence();
    }

    void testSingleSequence()
    {
      // "Test: Single sequence"
      Sequence sequence = _sequenceMachine.generateFromNumbers(vector<vector<UInt>>{ range(50) });
      vector<Real64> times = _feedAll(sequence);

      NTA_CHECK(times[0] < times[1]);
      NTA_CHECK(times[2] < times[1]);
      NTA_CHECK(times[2] < times[0]);
    }

    // ==============================
    // Helper functions
    // ==============================

    Real64 _feedOne(Sequence& sequence, Instance& instance, ComputeFunction& computeFn)
    {
      Timer timer(true); // auto start enabled

      for (vector<UInt> pattern : sequence)
      {
        if (pattern.size() == 0)
          instance.reset();
        else
          computeFn(pattern, instance);
      }

      timer.stop();

      return timer.getElapsed();
    }

    vector<Real64> _feedAll(Sequence& sequence, bool learn = true, int num = 1)
    {
      vector<Real64> times;
      Real64 elapsed;
      Sequence repeatedSequence;

      for (int i = 0; i < num; i++)
      {
        for (auto seq : sequence)
          repeatedSequence.push_back(seq);
      }

      elapsed = _feedOne(repeatedSequence, tm, tmComputeFn);
      times.push_back(elapsed);
      cout << "TM:\t" << elapsed << " s";

      elapsed = _feedOne(repeatedSequence, tp, tpComputeFn);
      times.push_back(elapsed);
      cout << "TP:\t" << elapsed << " s";

      elapsed = _feedOne(repeatedSequence, sp, tpComputeFn);
      times.push_back(elapsed);
      cout << "SP:\t" << elapsed << " s";

      return times;
    }
  };
};
