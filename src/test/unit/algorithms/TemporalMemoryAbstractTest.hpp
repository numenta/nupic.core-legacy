/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#ifndef NTA_TEMPORALMEMORYABSTRACTTEST_HPP
#define NTA_TEMPORALMEMORYABSTRACTTEST_HPP

/** @file
 * Definition of unit tests for Temporal Memory abstract test
 */

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/test/Tester.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace std;
using namespace nupic::utils;
using namespace nupic::algorithms::temporal_memory;

namespace nupic {

  class TemporalMemoryAbstractTest : public Tester
  {
  public:
    TemporalMemoryAbstractTest() { _verbosity = 1; };
    virtual ~TemporalMemoryAbstractTest() {};

    /**
    * Initialize verbosity level, and the Pattern and Sequence machines
    */
    virtual void setUp();

    /**
     * Initialize the Temporal Memory
     */
    virtual void init();

    /**
     * Feed a sequence(s) into the TM with learning enabled or disabled
     */
    virtual void _feedTM(Sequence& sequence, bool learn = true, int num = 1);

  protected:
    int _verbosity;

    TemporalMemory _tm;
    PatternMachine  _patternMachine;
    SequenceMachine _sequenceMachine;

  }; // of class AbstractTemporalMemoryTest

}; // of namespace nupic

#endif // of NTA_TEMPORALMEMORYABSTRACTTEST_HPP
