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

/** @file
 * Implementation of unit tests for TemporalMemory
 */

#ifndef NTA_TemporalMemoryTutorial_TEST
#define NTA_TemporalMemoryTutorial_TEST

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/math/StlIo.hpp>
#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

#include "TemporalMemoryAbstractTest.hpp"

using namespace std;


namespace nupic {

  class TemporalMemoryTutorialTest : public TemporalMemoryAbstractTest
  {
  public:
    ConsecutivePatternMachine patternMachine;

    // Run all appropriate tests
    virtual void RunTests();

    void testFirstOrder();
    void testHighOrder();
    void testHighOrderAlternating();
    void testEndlesslyRepeating();
    void testEndlesslyRepeatingWithNoNewSynapses();
    void testLongRepeatingWithNovelEnding();
    void testSingleEndlesslyRepeating();


    // ==============================
    // Overrides
    // ==============================

    virtual void setUp();
    virtual void init();

    virtual void _feedTM(Sequence& sequence, bool learn = true, int num = 1);


    // ==============================
    // Helper functions
    // ==============================

    void _showInput(Sequence& sequence, bool learn = false, int num = 1);

  }; // of class TemporalMemoryTutorialTest

}; // of namespace nupic

#endif // of NTA_TemporalMemoryTutorial_TEST
