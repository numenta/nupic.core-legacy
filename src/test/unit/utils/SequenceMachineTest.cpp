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
 * Implementation of Sequence Machine test
 */

#include <string>
#include <sstream>
#include <exception>
#include <vector>
#include <iterator>
#include <functional>

#include "SequenceMachineTest.hpp"

using namespace nupic;

bool SequenceMachineTest::check_pattern_eq(vector<UInt>& p1, vector<UInt>& p2)
{
  for (UInt i = 0; i < p1.size(); i++) {
    if (p1[i] != p2[i]) {
      return false;
    }
  }
  return true;
}

vector<UInt> SequenceMachineTest::get_pattern_diffs(vector<UInt>& p1, vector<UInt>& p2)
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

vector<UInt> SequenceMachineTest::get_pattern_intersections(vector<UInt>& p1, vector<UInt>& p2)
{
  vector<UInt> s1 = p1, s2 = p2;
  vector<UInt> diffs;

  // For the set_symmetric_difference algorithm to work, the source ranges must be ordered!    
  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());

  // Now that we have sorted ranges (i.e., containers), find the differences    
  set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(diffs));

  return diffs;
}

vector<UInt> SequenceMachineTest::get_pattern_union(vector<UInt>& p1, vector<UInt>& p2)
{
  vector<UInt> s1(p1), s2(p2);
  vector<UInt> combined;
  std::vector<UInt>::iterator it;

  sort(s1.begin(), s1.end());
  sort(s2.begin(), s2.end());

  combined.resize(s1.size()+s2.size());
  it = std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), combined.begin());
  combined.resize(it - combined.begin());

  return combined;
}

SequenceMachineTest::SequenceMachineTest()
{
  _patternMachine = ConsecutivePatternMachine();
  vector<UInt> w = { 5 };
  _patternMachine.initialize(100, w);

  _sequenceMachine = SequenceMachine(_patternMachine);

}

void SequenceMachineTest::RunTests()
{
  testGenerateFromNumbers();
  testAddSpatialNoise();
  testGenerateNumbers();
  testGenerateNumbersMultipleSequences();
  testGenerateNumbersWithShared();

}

void SequenceMachineTest::testGenerateFromNumbers()
{
  Sequence numbers({ range(0, 10), {}, range(10, 20) });

  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  NTA_CHECK(sequence.size() == 21);
  vector<UInt> seq1 = _patternMachine.get(0);
  vector<UInt> s1 = sequence[0];
  NTA_CHECK(check_pattern_eq(s1, seq1));

  vector<UInt> reset = _patternMachine.get(0);
  vector<UInt> s2 = sequence[10];
  NTA_CHECK(check_pattern_eq(s2, reset));

  vector<UInt> seq2 = _patternMachine.get(10);
  vector<UInt> s3 = sequence[11];
  NTA_CHECK(check_pattern_eq(s3, seq2));

}

void SequenceMachineTest::testAddSpatialNoise()
{
  PatternMachine patternMachine;
  vector<UInt> w = { 1000 };
  patternMachine.initialize(10000, w);

  SequenceMachine sequenceMachine = SequenceMachine(patternMachine);

  Sequence numbers({ range(0, 100), {} });
  vector<UInt> overlap;

  Sequence sequence = sequenceMachine.generateFromNumbers(numbers);
  Sequence noisy = sequenceMachine.addSpatialNoise(sequence, 0.5);

  vector<UInt> pattern = patternMachine.get(0);
  vector<UInt> noise = noisy[0];
  overlap = get_pattern_intersections(noise, pattern);
  NTA_CHECK((400 < overlap.size()) && (overlap.size() < 600));

  sequence = sequenceMachine.generateFromNumbers(numbers);
  noisy = sequenceMachine.addSpatialNoise(sequence, 0.0);

  pattern = patternMachine.get(0);
  noise = noisy[0];
  overlap = get_pattern_intersections(noise, pattern);
  NTA_CHECK((overlap.size() == 1000));

}

void SequenceMachineTest::testGenerateNumbers()
{
  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  vector<UInt> p = range(0, 100);
  vector<UInt> n = numbers[0];
  NTA_CHECK(check_pattern_eq(n, p) == false);

  vector<UInt> v = numbers[0];
  sort(v.begin(), v.end());
  NTA_CHECK(check_pattern_eq(v, p));
}

void SequenceMachineTest::testGenerateNumbersMultipleSequences()
{
  Sequence numbers = _sequenceMachine.generateNumbers(3, 100);
  vector<UInt> p;

  p = range(0, 100);
  vector<UInt> v0 = numbers[0];
  sort(v0.begin(), v0.end());
  NTA_CHECK(check_pattern_eq(v0, p));

  p = range(100, 200);
  vector<UInt> v1 = numbers[1];
  sort(v1.begin(), v1.end());
  NTA_CHECK(check_pattern_eq(v1, p));

  p = range(200, 300);
  vector<UInt> v2 = numbers[2];
  sort(v2.begin(), v2.end());
  NTA_CHECK(check_pattern_eq(v2, p));
}

void SequenceMachineTest::testGenerateNumbersWithShared()
{
  pair<int, int> sharedRange = { 20, 35 };
  Sequence numbers = _sequenceMachine.generateNumbers(3, 100, sharedRange);
  vector<UInt> shared = range(300, 315);

  vector<UInt> v0 = numbers[0];
  vector<UInt> s0(v0.begin() + 20, v0.begin() + 35);
  NTA_CHECK(check_pattern_eq(s0, shared));

  vector<UInt> v1 = numbers[1];
  vector<UInt> s1(v1.begin() + 20, v1.begin() + 35);
  NTA_CHECK(check_pattern_eq(s1, shared));

  vector<UInt> v2 = numbers[2];
  vector<UInt> s2(v2.begin() + 20, v2.begin() + 35);
  NTA_CHECK(check_pattern_eq(s2, shared));
}

