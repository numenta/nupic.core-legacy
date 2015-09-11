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
 * Utilities for generating and manipulating sequences, for use in
 * experimentation and tests.
 */

#include <vector>
#include <nupic/utils/SequenceMachine.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::utils;

SequenceMachine::SequenceMachine(PatternMachine& patternMachine, int seed)
{
  // Save member variables
  _patternMachine = patternMachine;

  // Initialize member variables
  _random = Random(seed);
}


// @param numSequences(int)   Number of sequences to return, separated by None
// @param sequenceLength(int)   Length of each sequence
// @param sharedRange(tuple) (start index, end index) indicating range of shared subsequence in each sequence(None if no shared subsequences)
// @return (list)Numbers representing sequences
Sequence SequenceMachine::generateNumbers(int numSequences, int sequenceLength, pair<int, int>sharedRange)
{
  Sequence numbers;
  vector<UInt> sharedNumbers;

  if (sharedRange.first >= 0)
  {
    int sharedLength = sharedRange.second - sharedRange.first;
    sharedNumbers = range(numSequences * sequenceLength, numSequences * sequenceLength + sharedLength);
  }

  vector<UInt> sequenceRange = range(numSequences);
  for (int i : sequenceRange)
  {
    int start = i * sequenceLength;

    vector<UInt> newNumbers = range(start, start + sequenceLength);

    std::random_shuffle(newNumbers.begin(), newNumbers.end());

    if (sharedRange.first >= 0)
      newNumbers.insert(newNumbers.begin() + sharedRange.first, sharedNumbers.begin(), sharedNumbers.end());

    numbers.push_back(newNumbers);
    numbers.push_back({});
  }

  return numbers;
}

// Generate a sequence from a list of numbers.
// Note : Any 'None' in the list of numbers is considered a reset.
//  @param numbers(list) List of numbers
//  @return (list)Generated sequence
Sequence SequenceMachine::generateFromNumbers(Sequence& numbers)
{
  Sequence sequence;

  for (vector<UInt> numberSequence : numbers.data)
  {
    if (numberSequence.size() == 0)
    {
      // Append the reset pattern
      sequence.push_back(numberSequence);
    }
    else
    {
      for (UInt patternIndex : numberSequence)
      {
        vector<UInt> pattern = _patternMachine.get(patternIndex);
        sequence.push_back(pattern);
      }
    }
  }

  return sequence;
}

// Add spatial noise to each pattern in the sequence.
//  @param sequence(list)  Sequence
//  @param amount(float) Amount of spatial noise
//  @return (list)Sequence with spatial noise
Sequence SequenceMachine::addSpatialNoise(Sequence& sequence, Real amount)
{
  Sequence newSequence;

  for (auto pattern : sequence.data)
  {
    vector<UInt> noise = _patternMachine.addNoise(pattern, amount);

    newSequence.push_back(noise);
  }

  return newSequence;
}

// Pretty print a sequence.
//  @param sequence(list) Sequence
//  @param verbosity(int)  Verbosity level
//  @return (string)Pretty - printed text
string SequenceMachine::prettyPrintSequence(Sequence& sequence, int verbosity)
{
  string text = "";

  vector<UInt> sequenceRange(nupic::utils::range(sequence.size()));
  for (auto i : sequenceRange)
  {
    vector<UInt> pattern = sequence[i];

    if (pattern.size() == 0)
      text += "<reset>";
    else
      text += _patternMachine.prettyPrintPattern(pattern, verbosity);

    if (i >= sequence.size() - 1)
      text += "\n";
  }

  return text;
}
