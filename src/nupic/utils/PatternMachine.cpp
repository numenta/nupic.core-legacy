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

#include <nupic/utils/PatternMachine.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::utils;
      
void PatternMachine::initialize(int n, vector<int>& w, int num, int seed)
{
  // Save member variables
  _n = n;
  _w = w;
  _num = num;

  // Initialize member variables
  _random = Random(seed);
  _patterns.clear();

  _generate();
}

// Return a pattern for a number.
//  @param number(int) Number of pattern
//  @return (set)Indices of on bits
const set<int>& PatternMachine::get(int number)
{
  //if not number in self._patterns:
  //  raise IndexError("Invalid number")

  return _patterns[number];
}

// Add noise to pattern.
//  @param bits(set)   Indices of on bits
//  @param amount(float) Probability of switching an on bit with a random bit
//  @return (set)Indices of on bits in noisy pattern
set<int> PatternMachine::addNoise(set<int> bits, Real amount)
{
  set<int> newBits;

  for (int bit : bits)
  {
    if (_random.getReal64() < amount)
      newBits.insert(_random.getUInt32(_n));
    else
      newBits.insert(bit);
  }

  return newBits;
}


// Return the set of pattern numbers that match a bit.
//  @param bit(int) Index of bit
//  @return (set)Indices of numbers
set<int> PatternMachine::numbersForBit(int bit)
{
  //if (bit >= _n)
  //  raise IndexError("Invalid bit")

  set<int> numbers;

  UInt index = 0;
  for (set<int> pattern : _patterns)
  {
    if (pattern.find(bit) != pattern.end())
      numbers.insert(index);
  }

  return numbers;
}

// Return a map from number to matching on bits,
// for all numbers that match a set of bits.
//   @param bits(set) Indices of bits
//   @return (dict)Mapping from number = > on bits.
vector<set<int>> PatternMachine::numberMapForBits(set<int> bits)
{
  vector<set<int>> numberMap;

  for (int bit : bits)
  {
    set<int> numbers = numbersForBit(bit);

    for (int number : numbers)
    {
      if (number >= 0)
        numberMap[number].insert(bit);
    }
  }

  return numberMap;
}


string PatternMachine::prettyPrintPattern(set<int> bits, int verbosity)
{
  string text = "";
  /*
  Pretty print a pattern.

  @param bits(set) Indices of on bits
  @param verbosity(int) Verbosity level

  @return (string)Pretty - printed text
  """
  numberMap = self.numberMapForBits(bits)

  numberList = []
  numberItems = sorted(numberMap.iteritems(),
  key = lambda(number, bits) : len(bits),
  reverse = True)

  for number, bits in numberItems :

  if verbosity > 2:
  strBits = [str(n) for n in bits]
  numberText = "{0} (bits: {1})".format(number, ",".join(strBits))
  elif verbosity > 1:
  numberText = "{0} ({1} bits)".format(number, len(bits))
  else:
  numberText = str(number)

  numberList.append(numberText)

  text += "[{0}]".format(", ".join(numberList))

  */
  return text;
}

// Generates set of random patterns.
void PatternMachine::_generate()
{
  /*
  vector<int> candidates = range(self._n);

  for (i in xrange(_num))
  {
  _random.shuffle(candidates);
  set<int> pattern = candidates[0:_getW()];
  _patterns[i] = pattern;
  }
  */
}

// Gets a value of `w` for use in generating a pattern.
int PatternMachine::_getW()
{
  /*
  vector<int> w = _w;

  if (w.size() > 1)
  return w[_random.getUInt32(w.size())];
  else
  if (w.size() == 1)
  return w[0];
  */
  return -1;
}

// Pattern machine class that generates patterns with 
// non-overlapping, consecutive on bits.
void ConsecutivePatternMachine::_generate()
{
  // Generates set of consecutive patterns.

  int n = _n;
  vector<int> w = _w;

  //assert type(w) is int, "List for w not supported"

  //for i in xrange(n / w) :
  //  pattern = set(xrange(i * w, (i + 1) * w))
  //  self._patterns[i] = pattern
}