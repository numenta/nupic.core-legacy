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
#include <nupic/algorithms/TemporalMemory.hpp>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>

using namespace std;
using namespace nupic::algorithms::temporal_memory;

namespace nupic {
  //class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, TemporalMemory) : pass

  class AbstractTemporalMemoryTest : Tester
  {
  public:
    AbstractTemporalMemoryTest() { _verbosity = 1; }
    virtual ~AbstractTemporalMemoryTest() {}

    // Run all appropriate tests
    virtual void RunTests() override;

    int _verbosity;

    TemporalMemory  _tm;
    PatternMachine  _patternMachine;
    SequenceMachine _sequenceMachine;

    void setUp()
    {
      //tm = None;
      //patternMachine = None;
      //sequenceMachine = None;
    }

    // Initialize Temporal Memory, and other member variables.
    // param overrides : overrides for default Temporal Memory parameters
    void init()
    {
      //params = self._computeTMParams(overrides);
      _tm = TemporalMemory();// **params);

      patternMachine = PatternMachine();
      sequenceMachine = SequenceMachine(patternMachine);
    }

    void feedTM(set<int> sequence, bool learn = true, int num = 1)
    {
      vector<set<int>> repeatedSequence;
      
      for (int i = 0; i < num; i++)
      {
        repeatedSequence.push_back(sequence[i]);
      }

      tm.mmClearHistory();

      for (set<int> pattern : repeatedSequence)
      {
        if (pattern[0] < 0)
          tm.reset();
        else
          tm.compute(pattern, learn);
      }
    }

    // ==============================
    // Helper functions
    // ==============================

    paramsType _computeTMParams(overrides)
    {
      params = dict(self.DEFAULT_TM_PARAMS);
      params.update(overrides or {});
      return params;
    }

  } // of class AbstractTemporalMemoryTest
}; // of namespace nupic
*/
