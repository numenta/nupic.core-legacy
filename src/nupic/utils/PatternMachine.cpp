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

#include <vector>
#include <iterator>
#include <functional>
#include <boost/range/algorithm.hpp>
#include <boost/range/irange.hpp>
#include <nupic/utils/PatternMachine.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::utils;

// Return a pattern for a given integer range.
// If the step argument is omitted, it defaults to 1. 
// If the start argument is omitted, it defaults to 0. 
// The full form returns a list of plain integers [start, start + step, start + 2 * step, ...].
// If step is positive, the last element is the largest start + i * step less than stop; 
// if step is negative, the last element is the smallest start + i * step greater than stop.
// step must not be zero (or else ValueError is raised).
vector<UInt> nupic::utils::range(int start, int stop, int step)
{
  vector<UInt> x;
  int numEntries = ceil((Real)(stop - start) / (Real)step);

  if (numEntries <= 0)
    return x;

  if (step == 1)
  {
    // irange expects to create a valid range, 
    // and this can only work with step == 1
    boost::range::copy(boost::irange(start, stop), std::back_inserter(x));
  }
  else
  {
    x.resize(numEntries);
    for (int j = 0, i = start; j < x.size(); i += step)
      x[j++] = i;
  }

  return x;
}

vector<UInt> nupic::utils::range(int start, int stop)
{
  return utils::range(start, stop, 1);
}

vector<UInt> nupic::utils::range(int stop)
{
  return utils::range(0, stop, 1);
}

void PatternMachine::initialize(int n, vector<UInt>& w, int num, int seed)
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
vector<UInt> PatternMachine::get(int number)
{
  if (number < 0 || number >= _patterns.size())
    throw runtime_error("Invalid index");

  return _patterns[number];
}

// Generates set of random patterns.
void PatternMachine::_generate()
{
  vector<UInt> candidates = range(0, _n, 1);

  _patterns.resize(_num);

  vector<UInt> xrange = range(_num);
  for (int i : xrange)
  {
    _random.shuffle(candidates.begin(), candidates.end());

    vector<UInt> pattern;
    Int w = _getW();

    pattern.resize(w);
    _Copy_n(candidates.begin(), w, pattern.begin());

    _patterns[i] = pattern;
  }
}

// Gets a value of `w` for use in generating a pattern.
int PatternMachine::_getW()
{
  vector<UInt> w = _w;

  if (w.size() > 1)
    return w[_random.getUInt32(w.size())];
  else
    if (w.size() == 1)
      return w[0];

  return -1;
}

// Add noise to pattern.
//  @param bits(set)   Indices of on bits
//  @param amount(float) Probability of switching an on bit with a random bit
//  @return (set)Indices of on bits in noisy pattern
vector<UInt> PatternMachine::addNoise(vector<UInt>& bits, Real amount)
{
  vector<UInt> newBits;

  for (int bit : bits)
  {
    if (_random.getReal64() < amount)
      newBits.push_back(_random.getUInt32(_n));
    else
      newBits.push_back(bit);
  }

  return newBits;
}

// Return the set of pattern numbers that match a bit.
//  @param bit(int) Index of bit
//  @return (set)Indices of numbers
vector<UInt> PatternMachine::numbersForBit(int bit)
{
  if (bit < 0 || bit >= _n)
    throw runtime_error("Invalid bit");

  vector<UInt> numbers;

  UInt index = 0;
  for (vector<UInt> pattern : _patterns)
  {
    if (std::find(pattern.begin(), pattern.end(), bit) != pattern.end())
      numbers.push_back(index);

    index++;
  }

  return numbers;
}

// Return a map from number to matching on bits,
// for all numbers that match a set of bits.
//   @param bits(set) Indices of bits
//   @return (dict)Mapping from number = > on bits.
std::map<UInt, vector<UInt>> PatternMachine::numberMapForBits(vector<UInt>& bits)
{
  std::map<UInt, vector<UInt>> numberMap;

  for (int bit : bits)
  {
    vector<UInt> numbers = numbersForBit(bit);

    for (UInt number : numbers)
      numberMap[number].push_back(bit);
  }

  return numberMap;
}

string PatternMachine::prettyPrintPattern(vector<UInt>& bits, int verbosity)
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

// vector<UInt> machine class that generates patterns with 
// non-overlapping, consecutive on bits.
void ConsecutivePatternMachine::_generate()
{
  // Generates set of consecutive patterns.
  int n = _n;
  vector<UInt> w = _w;

  if (w.size() != 1 && w[0] != 0)
    throw runtime_error("List for w not supported");

  vector<UInt> xrange = range(n / w[0]);
  _patterns.resize(xrange.size());

  for (int i : xrange)
  {
    vector<UInt> pattern = range(i * w[0], (i + 1)*w[0], 1);
    _patterns[i] = pattern;
  }
}
