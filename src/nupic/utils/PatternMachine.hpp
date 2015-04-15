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
* 
*/

#ifndef NTA_PATTERN_MACHINE_HPP
#define NTA_PATTERN_MACHINE_HPP

#include <string>
#include <map>
#include <vector>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/Random.hpp>

using namespace std;

namespace nupic {
  namespace utils {

    extern vector<UInt> range(int start, int stop, int step);
    extern vector<UInt> range(int start, int stop);
    extern vector<UInt> range(int stop);

    class PatternMachine
    {
    public:
      PatternMachine() {};

      virtual ~PatternMachine() {}

      // Create initial patterns
      void initialize(int n, vector<UInt>& w, int num = 100, int seed = 42);

      // Generates set of random patterns.
      virtual void _generate();

      // Return a pattern for a number.
      //  @param number(int) Number of pattern
      //  @return (set)Indices of on bits
      vector<UInt> get(int number);

      // Gets a value of `w` for use in generating a pattern.
      int _getW();

      // Add noise to pattern.
      //  @param bits(set)   Indices of on bits
      //  @param amount(float) Probability of switching an on bit with a random bit
      //  @return (set)Indices of on bits in noisy pattern
      vector<UInt> addNoise(vector<UInt>& bits, Real amount);

      // Return the set of pattern numbers that match a bit.
      //  @param bit(int) Index of bit
      //  @return (set)Indices of numbers
      vector<UInt> numbersForBit(int bit);

      // Return a map from number to matching on bits,
      // for all numbers that match a set of bits.
      //   @param bits(set) Indices of bits
      //   @return (dict)Mapping from number = > on bits.
      map<UInt, vector<UInt>> numberMapForBits(vector<UInt>& bits);

      string prettyPrintPattern(vector<UInt>& bits, int verbosity = 1);

      // Number of available bits in pattern
      int _n;

      // Number of on bits in pattern. If list, each pattern will have 
      // a `w` randomly selected from the list.
      vector<UInt> _w;

      // Number of available patterns
      int _num;
      vector<vector<UInt>> _patterns;

      Random _random;

    };

    class ConsecutivePatternMachine : public PatternMachine
    {
    public:
      // vector<UInt> machine class that generates patterns with 
      // non-overlapping, consecutive on bits.
      void _generate();

    };

  } // end namespace utils
} // end namespace nupic
#endif // NTA_PATTERN_MACHINE_HPP
