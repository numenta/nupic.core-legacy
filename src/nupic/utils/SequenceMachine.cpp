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
vector<int> SequenceMachine::generateNumbers(int numSequences, int sequenceLength, pair<int, int>sharedRange)
{
  vector<int> numbers, sharedNumbers;
/*
  if (sharedRange.first >= 0)
  {
    int sharedLength = sharedRange.second - sharedRange.first;
    //sharedNumbers = range(numSequences * sequenceLength, numSequences * sequenceLength + sharedLength);
    boost::copy(
      boost::counting_range(numSequences * sequenceLength, numSequences * sequenceLength + sharedLength),
      sharedNumbers);
  }

  for (int i : xrange(numSequences))
  {
    int start = i * sequenceLength;

    vector<int> newNumbers;
    boost::copy(boost::counting_range(start, start + sequenceLength), newNumbers.end());

    _random.shuffle(newNumbers);

    if (sharedRange.size() > 0)
      //newNumbers[sharedStart:sharedEnd] = sharedNumbers;
      boost::copy(newNumbers.begin() + sharedStart, sharedNumbers);

    numbers += newNumbers;
    numbers.push_back(-1);
  }
*/
  return numbers;
}

// Generate a sequence from a list of numbers.
// Note : Any `None` in the list of numbers is considered a reset.
//  @param numbers(list) List of numbers
//  @return (list)Generated sequence
vector<int> SequenceMachine::generateFromNumbers(vector<int>& numbers)
{
  vector<int> sequence;

  for (int number : numbers)
  {
    if (number < 0)
      sequence.push_back(number);
    else
    {
      Pattern& pattern = _patternMachine.get(number);
      sequence.insert(sequence.end(), pattern.begin(), pattern.end());
    }
  }

  return sequence;
}

// Add spatial noise to each pattern in the sequence.
//  @param sequence(list)  Sequence
//  @param amount(float) Amount of spatial noise
//  @return (list)Sequence with spatial noise
vector<int> SequenceMachine::addSpatialNoise(vector<int>& sequence, Real amount)
{
  vector<int> newSequence;
/*
  for (auto pattern : sequence)
  {
    if (pattern is not None)
      pattern = _patternMachine.addNoise(pattern, amount);

    newSequence.append(pattern);
  }
*/
  return newSequence;
}

// Pretty print a sequence.
//  @param sequence(list) Sequence
//  @param verbosity(int)  Verbosity level
//  @return (string)Pretty - printed text
string SequenceMachine::prettyPrintSequence(vector<int>& sequence, int verbosity)
{
  string text = "";
/*
  for (auto i : xrange(len(sequence)))
  {
    Pattern pattern = sequence[i];

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