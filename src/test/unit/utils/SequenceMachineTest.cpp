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

SequenceMachineTest::SequenceMachineTest()
{
  _patternMachine = ConsecutivePatternMachine();
  _patternMachine.initialize(100, vector<int>{ 5 });

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
  vector<int> numbers, range0, range1, range2;
  range0 = range(0, 10);
  range1 = vector<int>{ -1 };
  range2 = range(10, 19);
  
  numbers.insert(numbers.end(), range0.begin(), range0.end());
  numbers.insert(numbers.end(), range1.begin(), range1.end());
  numbers.insert(numbers.end(), range2.begin(), range2.end());

  vector<int> sequence = _sequenceMachine.generateFromNumbers(numbers);

//  self.assertEqual(len(sequence), 20)
//  self.assertEqual(sequence[0], self.patternMachine.get(0))
//  self.assertEqual(sequence[10], None)
//  self.assertEqual(sequence[11], self.patternMachine.get(10))
}

void SequenceMachineTest::testAddSpatialNoise()
{
  /*
    patternMachine = PatternMachine(10000, 1000, num = 100)
    sequenceMachine = SequenceMachine(patternMachine)
    numbers = range(0, 100)
    numbers.append(None)

    sequence = sequenceMachine.generateFromNumbers(numbers)
    noisy = sequenceMachine.addSpatialNoise(sequence, 0.5)

    overlap = len(noisy[0] & patternMachine.get(0))
    self.assertTrue(400 < overlap < 600)

    sequence = sequenceMachine.generateFromNumbers(numbers)
    noisy = sequenceMachine.addSpatialNoise(sequence, 0.0)

    overlap = len(noisy[0] & patternMachine.get(0))
    self.assertEqual(overlap, 1000)
  */
}

void SequenceMachineTest::testGenerateNumbers()
{
//  numbers = self.sequenceMachine.generateNumbers(1, 100)
//  self.assertEqual(numbers[-1], None)
//  self.assertEqual(len(numbers), 101)
//  self.assertFalse(numbers[:-1] == range(0, 100))
//  self.assertEqual(sorted(numbers[:-1]), range(0, 100))
}

void SequenceMachineTest::testGenerateNumbersMultipleSequences()
{
  //  numbers = self.sequenceMachine.generateNumbers(3, 100)
  //  self.assertEqual(len(numbers), 303)

  //  self.assertEqual(sorted(numbers[0:100]), range(0, 100))
  //  self.assertEqual(sorted(numbers[101:201]), range(100, 200))
  //  self.assertEqual(sorted(numbers[202:302]), range(200, 300))
}

void SequenceMachineTest::testGenerateNumbersWithShared()
{
  //  numbers = self.sequenceMachine.generateNumbers(3, 100, (20, 35))
  //  self.assertEqual(len(numbers), 303)

  //  shared = range(300, 315)
  //  self.assertEqual(numbers[20:35], shared)
  //  self.assertEqual(numbers[20 + 101:35 + 101], shared)
  //  self.assertEqual(numbers[20 + 202:35 + 202], shared)
}
