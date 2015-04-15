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
* ----------------------------------------------------------------------
*/

/** @file
* Utilities for generating and manipulating sequences, for use in
* experimentation and tests.
*/

#include <vector>

#include <nupic/utils/SequenceMachine.hpp>
//import numpy

using namespace std;
using namespace nupic;
using namespace nupic::utils;

SequenceMachine::SequenceMachine()
{
}

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

    _random.shuffle(newNumbers.begin(), newNumbers.end());

    if (sharedRange.first >= 0)
      newNumbers.insert(newNumbers.begin() + sharedRange.first, sharedNumbers.begin(), sharedNumbers.end());

    numbers.push_back(newNumbers);
  }

  return numbers;
}

// Generate a sequence from a list of numbers.
// Note : Any 'None' in the list of numbers is considered a reset.
//  @param numbers(list) List of numbers
//  @return (list)Generated sequence
Sequence SequenceMachine::generateFromNumbers(vector<vector<UInt>>& numbers)
{
  Sequence sequence;

  for (vector<UInt> numberSequence : numbers)
  {
    if (numberSequence.size() == 0)
      sequence.push_back(numberSequence);
    else
    {
      for (Int number : numberSequence)
        sequence.push_back(_patternMachine.get(number));
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

  for (auto pattern : sequence)
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
/*
  for (auto i : xrange(len(sequence)))
  {
    vector<UInt> pattern = sequence[i];

    if (pattern == None)
      text += "<reset>";
    if (i < sequence.size() - 1)
      text += "\n";
    else
      text += _patternMachine.prettyPrintPattern(pattern, verbosity);
  }
*/
  return text;
}
