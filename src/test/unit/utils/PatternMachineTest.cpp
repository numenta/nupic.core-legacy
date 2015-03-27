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
* Implementation of Pattern Machine test
*/

#include <string>
#include <sstream>
#include <exception>
#include <vector>
#include <iterator>
#include <functional>

#include "PatternMachineTest.hpp"

using namespace nupic;

void PatternMachineTest::RunTests()
{
  _patternMachine = PatternMachine();
  _patternMachine.initialize(10000, vector<int>{ 5 }, 50);

  testRange();
  testGet();
  testGetOutOfBounds();
  testAddNoise();
  testNumbersForBit();
  testNumbersForBitOutOfBounds();
  testNumberMapForBits();
  testWList();
  
  ConsecutivePatternMachineTest_setUp();
  ConsecutivePatternMachineTest_testGet();
  ConsecutivePatternMachineTest_testGetOutOfBounds();
}

bool PatternMachineTest::check_pattern_eq(Pattern& p1, Pattern& p2)
{
  for (UInt i = 0; i < p1.size(); i++) {
    if (p1[i] != p2[i]) {
      return false;
    }
  }
  return true;
}

Pattern PatternMachineTest::get_pattern_diffs(Pattern& p1, Pattern& p2)
{
  set<int> s1, s2;
  vector<int> diffs;

  s1.insert(p1.begin(), p1.end());
  s2.insert(p2.begin(), p2.end());

  // For the set_symmetric_difference algorithm to work, the source ranges must be ordered!    
  //sort(s1.begin(), s1.end());
  //sort(s2.begin(), s2.end());

  // Now that we have sorted ranges (i.e., containers), find the differences    
  set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(diffs));

  return diffs;
}

Pattern PatternMachineTest::get_pattern_union(Pattern& p1, Pattern& p2)
{
  vector<int> s1(p1), s2(p2);
  vector<int> combined;
  std::vector<int>::iterator it;

  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());
  combined.resize(s1.size());

  it = std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), combined.begin());
  combined.resize(it - combined.begin());

  return combined;
}

void PatternMachineTest::testRange()
{
  Pattern t, r;

  t = nupic::utils::range(10);
  r = Pattern{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(1, 11);
  r = Pattern{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0, 30, 5);
  r = Pattern{ 0, 5, 10, 15, 20, 25 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0, 10, 3);
  r = Pattern{ 0, 3, 6, 9 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0, -10, -1);
  r = Pattern{ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0);
  NTA_CHECK((t.size() == 0));

  t = nupic::utils::range(1, 0);
  NTA_CHECK((t.size() == 0));

}

void PatternMachineTest::testGet()
{
  Pattern patternA = _patternMachine.get(48);
  NTA_CHECK((patternA.size() == 5));

  Pattern patternB = _patternMachine.get(49);
  NTA_CHECK((patternB.size() == 5));

  Pattern diffs = get_pattern_diffs(patternA, patternB);
  NTA_CHECK((diffs.size() > 0));
}

void PatternMachineTest::testGetOutOfBounds()
{
  EXPECT_THROW(_patternMachine.get(50), runtime_error);
}

void PatternMachineTest::testAddNoise()
{
  PatternMachine patternMachine = PatternMachine();
  patternMachine.initialize(10000, vector<int>{ 1000 }, 1);

  Pattern diffs, noisy, pattern = patternMachine.get(0);

  noisy = patternMachine.addNoise(pattern, 0.0);
  diffs = get_pattern_diffs(pattern, noisy);
  NTA_CHECK((diffs.size() == 0));

  noisy = patternMachine.addNoise(pattern, 0.5);
  diffs = get_pattern_diffs(pattern, noisy);
//  NTA_CHECK((400 < diffs.size()) && (diffs.size() < 600));

  noisy = patternMachine.addNoise(pattern, 1.0);
  diffs = get_pattern_diffs(pattern, noisy);
//  NTA_CHECK((50 < diffs.size()) && (diffs.size() < 150));
}

void PatternMachineTest::testNumbersForBit()
{
  Pattern pattern = _patternMachine.get(49);

  for (int bit : pattern)
  {
    NTA_CHECK(_patternMachine.numbersForBit(bit) == Pattern{ 49 });
  }
}

void PatternMachineTest::testNumbersForBitOutOfBounds()
{
  EXPECT_THROW(_patternMachine.numbersForBit(10000), runtime_error);
}

void PatternMachineTest::testNumberMapForBits()
{
  Pattern pattern = _patternMachine.get(49);
  map<int, Pattern> numberMap = _patternMachine.numberMapForBits(pattern);

  NTA_CHECK(numberMap[49].size() > 0);
  NTA_CHECK(check_pattern_eq(numberMap[49], pattern));
}

void PatternMachineTest::testWList()
{
  vector<int> w = { 4, 7, 11 };

  PatternMachine _patternMachine;
  _patternMachine.initialize(100, w, 50);

  map<int, int> widths;

  Pattern r = nupic::utils::range(50);
  for (auto i : r)
  {
    Pattern pattern = _patternMachine.get(i);
    int width = pattern.size();
    NTA_CHECK(find(w.begin(), w.end(), width) != w.end());
    widths[pattern.size()] += 1;
  }
  for (auto i : w)
  {
    NTA_CHECK(widths[i] > 0);
  }
}

void PatternMachineTest::ConsecutivePatternMachineTest_setUp()
{
  ConsecutivePatternMachine patternMachine = ConsecutivePatternMachine();
  patternMachine.initialize(100, vector<int>{ 5 });

  _patternMachine = patternMachine;
}

void PatternMachineTest::ConsecutivePatternMachineTest_testGet()
{
  Pattern pattern = _patternMachine.get(18);
  NTA_CHECK(pattern.size() == 5);

  Pattern t, r;

  r = Pattern{ 90, 91, 92, 93, 94 };
  NTA_CHECK(check_pattern_eq(pattern, r));

  pattern = _patternMachine.get(19);
  NTA_CHECK(pattern.size() == 5);

  r = Pattern{ 95, 96, 97, 98, 99 };
  NTA_CHECK(check_pattern_eq(pattern, r));
}

void PatternMachineTest::ConsecutivePatternMachineTest_testGetOutOfBounds()
{
  EXPECT_THROW(_patternMachine.get(20), runtime_error);
}
