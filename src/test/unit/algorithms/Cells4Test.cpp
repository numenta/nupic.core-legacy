/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

#include <vector>

#include <nupic/algorithms/Cells4.hpp>
#include <gtest/gtest.h>

using namespace nupic::algorithms::Cells4;


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with removals,
 * inactive, and active synapses.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithRemovalsInMiddleAndEdges)
{
  std::vector<UInt> removedSrcCellIdxs {99, 77, 44};
  std::vector<UInt> inactiveSrcCellIdxs {99, 88, 77, 66, 55, 44};
  std::vector<UInt> inactiveSynapseIdxs { 1,  3,  7, 11, 12, 23};
  std::vector<UInt> activeSynapseIdxs {0, 2, 4, 6, 9, 10, 50, 60};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {99, 77, 44};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {88, 66, 55};
  const std::vector<UInt> expectedInactiveSynapseIdxs { 2,  9, 10};
  const std::vector<UInt> expectedActiveSynapseIdxs {0,  1, 3, 5, 7, 8, 47, 57};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);
  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with nothing
 * removed.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithNoRemovals)
{
  std::vector<UInt> removedSrcCellIdxs {};
  std::vector<UInt> inactiveSrcCellIdxs {99, 88, 77, 66, 55, 44};
  std::vector<UInt> inactiveSynapseIdxs { 1,  3,  7, 11, 12, 23};
  std::vector<UInt> activeSynapseIdxs {0, 2, 4, 8, 9, 50, 60};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {99, 88, 77, 66, 55, 44};
  const std::vector<UInt> expectedInactiveSynapseIdxs { 1,  3,  7, 11, 12, 23};
  const std::vector<UInt> expectedActiveSynapseIdxs {0, 2, 4, 8, 9, 50, 60};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);

  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment without inactive
 * synapses.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithNoInactiveSynapses)
{
  std::vector<UInt> removedSrcCellIdxs {};
  std::vector<UInt> inactiveSrcCellIdxs {};
  std::vector<UInt> inactiveSynapseIdxs {};
  std::vector<UInt> activeSynapseIdxs {0, 2, 4, 8, 9, 50, 60};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSynapseIdxs {};
  const std::vector<UInt> expectedActiveSynapseIdxs {0, 2, 4, 8, 9, 50, 60};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);

  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with no
 * active synapses.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithNoActiveSynapses)
{
  std::vector<UInt> removedSrcCellIdxs {99, 77, 44};
  std::vector<UInt> inactiveSrcCellIdxs {99, 88, 77, 66, 55, 44};
  std::vector<UInt> inactiveSynapseIdxs { 1,  3,  7, 11, 12, 23};
  std::vector<UInt> activeSynapseIdxs {};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {99, 77, 44};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {88, 66, 55};
  const std::vector<UInt> expectedInactiveSynapseIdxs { 2,  9, 10};
  const std::vector<UInt> expectedActiveSynapseIdxs {};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);

  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with one
 * removal in middle, and with innactive, and active synapses.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithSingleRemovalInMiddle)
{
  std::vector<UInt> removedSrcCellIdxs {77};
  std::vector<UInt> inactiveSrcCellIdxs {99, 88, 77, 66, 55, 44};
  std::vector<UInt> inactiveSynapseIdxs { 1,  3,  7, 11, 12, 23};
  std::vector<UInt> activeSynapseIdxs {0, 2, 4, 8, 9, 50, 60};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {77};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {99, 88, 66, 55, 44};
  const std::vector<UInt> expectedInactiveSynapseIdxs { 1,  3, 10, 11, 22};
  const std::vector<UInt> expectedActiveSynapseIdxs {0, 2, 4, 7, 8, 49, 59};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);
  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with one
 * removal, single inactive synapse, and active synapses out of range.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithSingleRemovalSingleInactiveAndSingleActiveOutOfRange)
{
  std::vector<UInt> removedSrcCellIdxs {77};
  std::vector<UInt> inactiveSrcCellIdxs {77};
  std::vector<UInt> inactiveSynapseIdxs {7};
  std::vector<UInt> activeSynapseIdxs {0};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {77};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSynapseIdxs {};
  const std::vector<UInt> expectedActiveSynapseIdxs {0};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);
  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}


/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with one
 * removal, single inactive synapse, and single active in range.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithSingleRemovalSingleInactiveAndSingleActiveInRange)
{
  std::vector<UInt> removedSrcCellIdxs {77};
  std::vector<UInt> inactiveSrcCellIdxs {77};
  std::vector<UInt> inactiveSynapseIdxs {7};
  std::vector<UInt> activeSynapseIdxs {8};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {77};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSynapseIdxs {};
  const std::vector<UInt> expectedActiveSynapseIdxs {7};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);
  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}
/*
 * Test Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment with no
 * synapses.
 */
TEST(Cells4Test, fixupIndexesInAdaptSegmentWithNoSynapses)
{
  std::vector<UInt> removedSrcCellIdxs {};
  std::vector<UInt> inactiveSrcCellIdxs {};
  std::vector<UInt> inactiveSynapseIdxs {};
  std::vector<UInt> activeSynapseIdxs {};

  const std::vector<UInt> expectedRemovedSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSrcCellIdxs {};
  const std::vector<UInt> expectedInactiveSynapseIdxs {};
  const std::vector<UInt> expectedActiveSynapseIdxs {};

  Cells4::_fixupIndexesAfterSynapseRemovalsInAdaptSegment(removedSrcCellIdxs,
                                                          inactiveSrcCellIdxs,
                                                          inactiveSynapseIdxs,
                                                          activeSynapseIdxs);

  ASSERT_EQ(expectedRemovedSrcCellIdxs, removedSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSrcCellIdxs, inactiveSrcCellIdxs);
  ASSERT_EQ(expectedInactiveSynapseIdxs, inactiveSynapseIdxs);
  ASSERT_EQ(expectedActiveSynapseIdxs, activeSynapseIdxs);
}
