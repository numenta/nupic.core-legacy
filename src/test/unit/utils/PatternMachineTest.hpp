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
* vector<UInt> Machine unit tests
*/

#ifndef NTA_PATTERN_MACHINE_TEST_HPP
#define NTA_PATTERN_MACHINE_TEST_HPP

#include <nupic/test/Tester.hpp>

#include <nupic/utils/PatternMachine.hpp>

using namespace nupic::utils;

namespace nupic
{
  class PatternMachineTest : public Tester
  {
  public:
    virtual ~PatternMachineTest() {};
    virtual void RunTests() override;

    PatternMachine _patternMachine;

    bool check_pattern_eq(vector<UInt>& p1, vector<UInt>& p2);
    vector<UInt> get_pattern_diffs(vector<UInt>& p1, vector<UInt>& p2);
    vector<UInt> get_pattern_union(vector<UInt>& p1, vector<UInt>& p2);
    vector<UInt> get_pattern_intersections(vector<UInt>& p1, vector<UInt>& p2);

    void testRange();
    void testGet();
    void testGetOutOfBounds();
    void testAddNoise();
    void testNumbersForBit();
    void testNumbersForBitOutOfBounds();
    void testNumberMapForBits();
    void testWList();

    void ConsecutivePatternMachineTest_setUp();
    void ConsecutivePatternMachineTest_testGet();
    void ConsecutivePatternMachineTest_testGetOutOfBounds();
  };
}

#endif // NTA_PATTERN_MACHINE_TEST_HPP
