/*
 * Copyright 2016 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Implementation of unit tests for Segment
 */

#include <gtest/gtest.h>
#include <nupic/algorithms/Segment.hpp>
#include <set>

using namespace nupic::algorithms::Cells4;
using namespace std;

void setUpSegment(Segment &segment, vector<UInt> &inactiveSegmentIndices,
                  vector<UInt> &activeSegmentIndices,
                  vector<UInt> &activeSynapseIndices,
                  vector<UInt> &inactiveSynapseIndices) {
  vector<double> permanences = {0.2, 0.9, 0.9, 0.7, 0.4,  // active synapses
                                0.8, 0.1, 0.2, 0.3, 0.2}; // inactive synapses

  set<UInt> srcCells;
  for (UInt i = 0; i < permanences.size(); i++) {
    srcCells.clear();
    srcCells.insert(i);

    segment.addSynapses(srcCells, permanences[i], 0.5);

    if (i < 5) {
      inactiveSegmentIndices.push_back(i);
      inactiveSynapseIndices.push_back(0);
    } else {
      activeSegmentIndices.push_back(i);
      activeSynapseIndices.push_back(0);
    }
  }
}

/*
 * Test that synapses are removed from inactive first even when there
 * are active synapses with lower permanence.
 */
TEST(SegmentTest, freeNSynapsesInactiveFirst) {
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;
  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment, inactiveSegmentIndices, activeSegmentIndices,
               activeSynapseIndices, inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(2, inactiveSynapseIndices, inactiveSegmentIndices,
                        activeSynapseIndices, activeSegmentIndices, removed, 0,
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
TEST(SegmentTest, freeNSynapsesActiveFallback) {
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;

  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment, inactiveSegmentIndices, activeSegmentIndices,
               activeSynapseIndices, inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(6, inactiveSynapseIndices, inactiveSegmentIndices,
                        activeSynapseIndices, activeSegmentIndices, removed, 0,
                        10, 1.0);

  vector<UInt> removed_expected = {0, 1, 2, 3, 4, 6};
  sort(removed.begin(), removed.end());
  ASSERT_EQ(removed, removed_expected);
}

/*
 * Test that removal respects insertion order (stable sort of permanences).
 */
TEST(SegmentTest, freeNSynapsesStableSort) {
  Segment segment;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;

  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;
  vector<UInt> removed;

  setUpSegment(segment, inactiveSegmentIndices, activeSegmentIndices,
               activeSynapseIndices, inactiveSynapseIndices);

  ASSERT_EQ(segment.size(), 10);

  segment.freeNSynapses(7, inactiveSynapseIndices, inactiveSegmentIndices,
                        activeSynapseIndices, activeSegmentIndices, removed, 0,
                        10, 1.0);

  vector<UInt> removed_expected = {0, 1, 2, 3, 4, 6, 7};
  sort(removed.begin(), removed.end());
  ASSERT_EQ(removed, removed_expected);
}

/**
 * Test operator '=='
 */
TEST(SegmentTest, testEqualsOperator) {
  Segment segment1;
  Segment segment2;

  vector<UInt> inactiveSegmentIndices;
  vector<UInt> activeSegmentIndices;
  vector<UInt> activeSynapseIndices;
  vector<UInt> inactiveSynapseIndices;

  setUpSegment(segment1, inactiveSegmentIndices, activeSegmentIndices,
               activeSynapseIndices, inactiveSynapseIndices);
  ASSERT_TRUE(segment1 != segment2);
  setUpSegment(segment2, inactiveSegmentIndices, activeSegmentIndices,
               activeSynapseIndices, inactiveSynapseIndices);
  ASSERT_TRUE(segment1 == segment2);
}