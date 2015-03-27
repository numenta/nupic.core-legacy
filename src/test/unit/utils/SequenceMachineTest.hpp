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
 * Sequence Machine unit tests
 */

#ifndef NTA_SEQUENCE_MACHINE_TEST_HPP
#define NTA_SEQUENCE_MACHINE_TEST_HPP

#include <nupic/test/Tester.hpp>

#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>

using namespace nupic::utils;

namespace nupic
{
  class SequenceMachineTest : public Tester
  {
  public:
    SequenceMachineTest();
    virtual ~SequenceMachineTest() {};
    virtual void RunTests() override;

    PatternMachine _patternMachine;
    SequenceMachine _sequenceMachine;

    bool check_pattern_eq(Pattern& p1, Pattern& p2);

    void testGenerateFromNumbers();
    void testAddSpatialNoise();
    void testGenerateNumbers();
    void testGenerateNumbersMultipleSequences();
    void testGenerateNumbersWithShared();
  };
}

#endif // NTA_SEQUENCE_MACHINE_TEST_HPP
