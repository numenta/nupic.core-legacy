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
* Definitions for unit tests for TemporalMemory & MonitorMixin classes
*/

#ifndef NTA_MonitorMixin_TEST
#define NTA_MonitorMixin_TEST

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>
#include <nupic/test/Tester.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/monitor_mixin/MonitorMixin.hpp>

#include "../TemporalMemoryAbstractTest.hpp"

using namespace std;

namespace nupic {

  class MonitorMixinTest : public Tester
  {
  public:
    MonitorMixinTest() {};
    virtual ~MonitorMixinTest() {}

    // Run all appropriate tests
    virtual void RunTests() override;

  private:
    void setUp();

    virtual void testFeedSequence();
    virtual void testClearHistory();
    virtual void testSequencesMetrics();

    void _generateSequence();
    void _feedSequence(utils::Sequence& sequence, string sequenceLabel = "");

    utils::PatternMachine _patternMachine;
    utils::SequenceMachine _sequenceMachine;
    TemporalMemoryMonitorMixin _tm;

  }; // end class MonitorMixinTest

} // end namespace nupic

#endif // NTA_MonitorMixin_TEST
