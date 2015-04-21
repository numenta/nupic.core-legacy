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
* Definition of unit tests for SpatialPooler
*/

#ifndef NTA_TemporalMemoryPerformanceTest_TEST
#define NTA_TemporalMemoryPerformanceTest_TEST

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/monitor_mixin/Instance.hpp>

#include <nupic/os/Timer.hpp>

#include <nupic/test/Tester.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

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

  // ==============================
  //  Tests
  // ==============================

  class TemporalMemoryPerformanceTest : public Tester
  {
  public:
    SpatialPoolerInstance sp;
    Cells4Instance tp;
    TemporalMemoryInstance tm;
    PatternMachine  _patternMachine;
    SequenceMachine _sequenceMachine;
    bool _learn;

    void setUp();

    // Run all appropriate tests
    virtual void RunTests();

    void testSingleSequence();

    // ==============================
    // Helper functions
    // ==============================

    Real64 _feedOne(Sequence& sequence, Instance& instance, ComputeFunction& computeFn);

    vector<Real64> _feedAll(Sequence& sequence, bool learn = true, int num = 1);

  };

}; // of namespace nupic

#endif // NTA_TemporalMemoryPerformanceTest_TEST
