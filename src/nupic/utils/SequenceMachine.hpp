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

#ifndef NTA_SEQUENCE_MACHINE_HPP
#define NTA_SEQUENCE_MACHINE_HPP

#include <vector>

#include <nupic/types/Types.hpp>
#include <nupic/math/Math.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/utils/PatternMachine.hpp>
//import numpy

using namespace std;
using namespace nupic;

namespace nupic {
  namespace utils {

    // Base sequence machine class.

    class SequenceMachine
    {
    private:

      PatternMachine _patternMachine;
      Random _random;

    public:

      SequenceMachine(const PatternMachine& patternMachine, int seed = 42);


      // @param numSequences(int)   Number of sequences to return, separated by None
      // @param sequenceLength(int)   Length of each sequence
      // @param sharedRange(tuple) (start index, end index) indicating range of shared subsequence in each sequence(None if no shared subsequences)
      // @return (list)Numbers representing sequences
      vector<int> generateNumbers(int numSequences, int sequenceLength, pair<int, int>sharedRange = { -1,-1 });

      // Generate a sequence from a list of numbers.
      // Note : Any `None` in the list of numbers is considered a reset.
      //  @param numbers(list) List of numbers
      //  @return (list)Generated sequence
      vector<int> generateFromNumbers(vector<int> numbers);

      // Add spatial noise to each pattern in the sequence.
      //  @param sequence(list)  Sequence
      //  @param amount(float) Amount of spatial noise
      //  @return (list)Sequence with spatial noise
      vector<int> addSpatialNoise(vector<int> sequence, Real amount);

      // Pretty print a sequence.
      //  @param sequence(list) Sequence
      //  @param verbosity(int)  Verbosity level
      //  @return (string)Pretty - printed text
      string prettyPrintSequence(vector<int> sequence, int verbosity = 1);
    };

  } // of namespace utils
} // of namespace nupic    

#endif // NTA_SEQUENCE_MACHINE_HPP