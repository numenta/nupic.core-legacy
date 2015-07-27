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
 * Implementation of unit tests for SpatialPooler
 */

#include "TemporalMemoryAbstractTest.hpp"


void TemporalMemoryAbstractTest::setUp()
{
}

// Initialize Temporal Memory, and other member variables.
void TemporalMemoryAbstractTest::init()
{
  // param overrides : overrides for default Temporal Memory parameters
  //params = self._computeTMParams(overrides);

  _tm = TemporalMemory();// **params);
  _tm.initialize({ 100 }, 1, 11, 0.8, 0.7, 11, 11, 0.4, 0.0, 42);
}

void TemporalMemoryAbstractTest::_feedTM(Sequence sequence, bool learn, int num)
{
  Sequence repeatedSequence;
      
  for (int i = 0; i < num; i++)
  {
    repeatedSequence += sequence;
  }

  for (vector<UInt> pattern : repeatedSequence.data)
  {
    if (pattern.size() == 0)
      _tm.reset();
    else
    {
      _tm.compute((UInt)pattern.size(), &pattern[0], learn);
    }
  }
}
