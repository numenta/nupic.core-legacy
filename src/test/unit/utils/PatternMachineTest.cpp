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
  _patternMachine.initialize(10000, vector<UInt>{ 5 }, 50);

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

bool PatternMachineTest::check_pattern_eq(vector<UInt>& p1, vector<UInt>& p2)
{
  for (UInt i = 0; i < p1.size(); i++) {
    if (p1[i] != p2[i]) {
      return false;
    }
  }
  return true;
}

vector<UInt> PatternMachineTest::get_pattern_diffs(vector<UInt>& p1, vector<UInt>& p2)
{
  vector<UInt> s1 = p1, s2 = p2;
  vector<UInt> diffs;

  // For the set_symmetric_difference algorithm to work, the source ranges must be ordered!    
  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());

  // Now that we have sorted ranges (i.e., containers), find the differences    
  set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(diffs));

  return diffs;
}

vector<UInt> PatternMachineTest::get_pattern_intersections(vector<UInt>& p1, vector<UInt>& p2)
{
  vector<UInt> s1 = p1, s2 = p2;
  vector<UInt> diffs;

  // For the set_symmetric_difference algorithm to work, the source ranges must be ordered!    
  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());

  // Now that we have sorted ranges (i.e., containers), find the differences    
  set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(diffs));

  return diffs;
}

vector<UInt> PatternMachineTest::get_pattern_union(vector<UInt>& p1, vector<UInt>& p2)
{
  vector<UInt> s1(p1), s2(p2);
  vector<UInt> combined;
  std::vector<UInt>::iterator it;

  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());

  combined.resize(s1.size() + s2.size());
  it = std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), combined.begin());
  combined.resize(it - combined.begin());

  return combined;
}

void PatternMachineTest::testRange()
{
  vector<UInt> t, r;

  t = nupic::utils::range(10);
  r = vector<UInt>{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(1, 11);
  r = vector<UInt>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0, 30, 5);
  r = vector<UInt>{ 0, 5, 10, 15, 20, 25 };
  NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0, 10, 3);
  r = vector<UInt>{ 0, 3, 6, 9 };
  NTA_CHECK(check_pattern_eq(t, r));

  // This is a valid range
  t = nupic::utils::range(0, -10, -1);
  // An invalid range for patterns
  //r = vector<UInt>{ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 };
  //NTA_CHECK(check_pattern_eq(t, r));

  t = nupic::utils::range(0);
  NTA_CHECK((t.size() == 0));

  t = nupic::utils::range(1, 0);
  NTA_CHECK((t.size() == 0));

}

void PatternMachineTest::testGet()
{
  vector<UInt> patternA = _patternMachine.get(48);
  NTA_CHECK((patternA.size() == 5));

  vector<UInt> patternB = _patternMachine.get(49);
  NTA_CHECK((patternB.size() == 5));

  vector<UInt> diffs = get_pattern_diffs(patternA, patternB);
  NTA_CHECK((diffs.size() > 0));
}

void PatternMachineTest::testGetOutOfBounds()
{
  EXPECT_THROW(_patternMachine.get(50), runtime_error);
}

void PatternMachineTest::testAddNoise()
{
  PatternMachine patternMachine = PatternMachine();
  patternMachine.initialize(10000, vector<UInt>{ 1000 }, 1);

  vector<UInt> diffs, noisy, pattern;

  pattern = patternMachine.get(0);

  noisy = patternMachine.addNoise(pattern, 0.0);
  diffs = get_pattern_intersections(pattern, noisy);
  NTA_CHECK((diffs.size() == 0));

  noisy = patternMachine.addNoise(pattern, 0.5);
  diffs = get_pattern_intersections(pattern, noisy);
  NTA_CHECK(diffs.size() > 500);

  noisy = patternMachine.addNoise(pattern, 1.0);
  diffs = get_pattern_intersections(pattern, noisy);
  NTA_CHECK(diffs.size() > 750);
}

void PatternMachineTest::testNumbersForBit()
{
  vector<UInt> pattern = _patternMachine.get(49);

  for (int bit : pattern)
  {
    NTA_CHECK(_patternMachine.numbersForBit(bit) == vector<UInt>{ 49 });
  }
}

void PatternMachineTest::testNumbersForBitOutOfBounds()
{
  EXPECT_THROW(_patternMachine.numbersForBit(10000), runtime_error);
}

void PatternMachineTest::testNumberMapForBits()
{
  vector<UInt> pattern = _patternMachine.get(49);
  map<UInt, vector<UInt>> numberMap = _patternMachine.numberMapForBits(pattern);

  NTA_CHECK(numberMap[49].size() > 0);
  NTA_CHECK(check_pattern_eq(numberMap[49], pattern));
}

void PatternMachineTest::testWList()
{
  vector<UInt> w = { 4, 7, 11 };

  PatternMachine _patternMachine;
  _patternMachine.initialize(100, w, 50);

  map<int, int> widths;

  vector<UInt> r = nupic::utils::range(50);
  for (auto i : r)
  {
    vector<UInt> pattern = _patternMachine.get(i);
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
  patternMachine.initialize(100, vector<UInt>{ 5 });

  _patternMachine = patternMachine;
}

void PatternMachineTest::ConsecutivePatternMachineTest_testGet()
{
  vector<UInt> pattern = _patternMachine.get(18);
  NTA_CHECK(pattern.size() == 5);

  vector<UInt> t, r;

  r = vector<UInt>{ 90, 91, 92, 93, 94 };
  NTA_CHECK(check_pattern_eq(pattern, r));

  pattern = _patternMachine.get(19);
  NTA_CHECK(pattern.size() == 5);

  r = vector<UInt>{ 95, 96, 97, 98, 99 };
  NTA_CHECK(check_pattern_eq(pattern, r));
}

void PatternMachineTest::ConsecutivePatternMachineTest_testGetOutOfBounds()
{
  EXPECT_THROW(_patternMachine.get(20), runtime_error);
}
