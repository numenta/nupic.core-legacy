/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for Segment
 */

#include <nupic/algorithms/Segment.hpp>
#include <gtest/gtest.h>
#include <set>

using namespace nupic::algorithms::Cells4;
using namespace std;


void setUpSegment(Segment &segment,
                  vector<UInt> &inactiveSegmentIndices,
                  vector<UInt> &activeSegmentIndices,
                  vector<UInt> &activeSynapseIndices,
                  vector<UInt> &inactiveSynapseIndices)
{
  vector<double> permanences = {0.2, 0.9, 0.9, 0.7, 0.4, // active synapses
                                0.8, 0.1, 0.2, 0.3, 0.2}; // inactive synapses

  set<UInt> srcCells;
  for (UInt i = 0; i < permanences.size(); i++)
  {
    srcCells.clear();
    srcCells.insert(i);

    segment.addSynapses(srcCells, permanences[i], 0.5);

    if (i < 5)
    {
      inactiveSegmentIndices.push_back(i);
      inactiveSynapseIndices.push_back(0);
    }
    else
    {
      activeSegmentIndices.push_back(i);
      activeSynapseIndices.push_back(0);
    }
  }
}

/*
* Test that synapses are removed from inactive first even when there
* are active synapses with lower permanence.
*/
TEST(SegmentTest, freeNSynapsesInactiveFirst)
{
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;
  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment,
               inactiveSegmentIndices,
               activeSegmentIndices,
               activeSynapseIndices,
               inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(2,
                        inactiveSynapseIndices,
                        inactiveSegmentIndices,
                        activeSynapseIndices,
                        activeSegmentIndices,
                        removed, 0,
                        10, 1.0);

  ASSERT_EQ(segment.size(), 8);

  vector<UInt> removed_expected = {0, 4};
  sort(removed.begin(), removed.end());
  ASSERT_EQ(removed, removed_expected);
}

/*
* Test that active synapses are removed once all inactive synapses are
* exhausted.
*/
TEST(SegmentTest, freeNSynapsesActiveFallback)
{
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;

  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment,
               inactiveSegmentIndices,
               activeSegmentIndices,
               activeSynapseIndices,
               inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(6,
                        inactiveSynapseIndices,
                        inactiveSegmentIndices,
                        activeSynapseIndices,
                        activeSegmentIndices,
                        removed, 0,
                        10, 1.0);

  vector<UInt> removed_expected = {0, 1, 2, 3, 4, 6};
  sort(removed.begin(), removed.end());
  ASSERT_EQ(removed, removed_expected);
}


/*
* Test that removal respects insertion order (stable sort of permanences).
*/
TEST(SegmentTest, freeNSynapsesStableSort)
{
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;

  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment,
               inactiveSegmentIndices,
               activeSegmentIndices,
               activeSynapseIndices,
               inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(7,
                        inactiveSynapseIndices,
                        inactiveSegmentIndices,
                        activeSynapseIndices,
                        activeSegmentIndices,
                        removed, 0,
                        10, 1.0);

  vector<UInt> removed_expected = {0, 1, 2, 3, 4, 6, 7};
  sort(removed.begin(), removed.end());
  ASSERT_EQ(removed, removed_expected);
}
