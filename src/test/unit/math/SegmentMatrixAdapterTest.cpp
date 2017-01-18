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
 * ----------------------------------------------------------------------
 */

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SegmentMatrixAdapter.hpp>
#include "gtest/gtest.h"

using std::vector;
using nupic::SegmentMatrixAdapter;
using nupic::SparseMatrix;
using nupic::UInt32;

namespace {

  /**
   * The SparseMatrix should contain one row for each added segment.
   */
  TEST(SegmentMatrixAdapterTest, addRows)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);
    EXPECT_EQ(0, ssm.matrix.nRows());

    ssm.createSegment(42);
    EXPECT_EQ(1, ssm.matrix.nRows());

    UInt32 cells[] = {42, 43, 44};
    UInt32 segmentsOut[3];
    ssm.createSegments(cells, cells + 3, segmentsOut);
    EXPECT_EQ(4, ssm.matrix.nRows());
  }

  /**
   * When you destroy a segment and then create a segment, the number of rows in
   * the SparseMatrix should stay constant.
   *
   * This test doesn't prescribe whether the SegmentMatrixAdapter should
   * accomplish this by keeping a list of "destroyed segments" or by simply
   * removing rows from the SparseMatrix.
   */
  TEST(SegmentMatrixAdapterTest, noRowLeaks)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);

    // Create 5 segments
    UInt32 cells1[] = {42, 43, 44, 45, 46};
    vector<UInt32> created(5);
    ssm.createSegments(cells1, cells1 + 5, created.data());
    ASSERT_EQ(5, ssm.matrix.nRows());

    // Destroy 3 segments, covering both destroy APIs
    ssm.destroySegment(created[1]);

    vector<UInt32> toDestroy = {created[2], created[3]};
    ssm.destroySegments(toDestroy.begin(), toDestroy.end());

    // Create 4 segments, covering both create APIs, and making sure
    // createSegments has to reuse destroyed segments *and* add rows in one
    // call.
    ssm.createSegment(50);

    UInt32 cells2[] = {51, 52, 53};
    UInt32 segmentsOut[3];
    ssm.createSegments(cells2, cells2 + 3, segmentsOut);

    EXPECT_EQ(6, ssm.matrix.nRows());
  }

  /**
   * Prepare:
   * - Cell that has multiple segments
   * - Cell that had multiple segments, then lost some of them
   * - Cell that has a single segment
   * - Cell that has had segments, then lost them
   * - Cell that has never had a segment
   *
   * Use both create APIs and both destroy APIs.
   *
   * Verify that getSegmentCounts gets the up-to-date count for each.
   */
  TEST(SegmentMatrixAdapterTest, getSegmentCounts)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);

    ssm.createSegment(42);

    vector<UInt32> cells = {42, 43, 44, 45};
    vector<UInt32> created(cells.size());
    ssm.createSegments(cells.begin(), cells.end(), created.data());

    ssm.createSegment(43);

    vector<UInt32> destroy = {created[1]};
    ssm.destroySegments(destroy.begin(), destroy.end());

    ssm.destroySegment(created[3]);

    vector<UInt32> queriedCells = {42, 43, 44, 45, 46};
    vector<UInt32> counts(queriedCells.size());
    ssm.getSegmentCounts(queriedCells.begin(), queriedCells.end(),
                         counts.begin());

    vector<UInt32> expected = {2, 1, 1, 0, 0};
    EXPECT_EQ(expected, counts);
  }

  TEST(SegmentMatrixAdapterTest, sortSegmentsByCell)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);

    UInt32 segment1 = ssm.createSegment(42);
    UInt32 segment2 = ssm.createSegment(41);
    UInt32 segment3 = ssm.createSegment(49);
    UInt32 segment4 = ssm.createSegment(45);
    UInt32 segment5 = ssm.createSegment(0);
    UInt32 segment6 = ssm.createSegment(2047);
    const vector<UInt32> sorted = {segment5,
                                   segment2,
                                   segment1,
                                   segment4,
                                   segment3,
                                   segment6};

    vector<UInt32> mySegments = {segment1,
                                 segment2,
                                 segment3,
                                 segment4,
                                 segment5,
                                 segment6};
    ssm.sortSegmentsByCell(mySegments.begin(), mySegments.end());

    EXPECT_EQ(sorted, mySegments);
  }

  TEST(SegmentMatrixAdapterTest, filterSegmentsByCell)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);

    // Don't create them in order -- we don't want the segment numbers
    // to be ordered in a meaningful way.

    // Shuffled
    // {42, 42, 42, 43, 46, 47, 48}
    const vector<UInt32> cellsWithSegments =
      {47, 42, 46, 43, 42, 48, 42};


    vector<UInt32> createdSegments(cellsWithSegments.size());
    ssm.createSegments(cellsWithSegments.begin(),
                       cellsWithSegments.end(),
                       createdSegments.begin());
    ssm.sortSegmentsByCell(createdSegments.begin(), createdSegments.end());

    // Include everything
    const vector<UInt32> everything = {42, 42, 42, 43, 46, 47, 48};
    EXPECT_EQ(createdSegments,
              ssm.filterSegmentsByCell(
                createdSegments.begin(), createdSegments.end(),
                everything.begin(), everything.end()));

    // Subset, one cell with multiple segments
    const vector<UInt32> subset1 = {42, 43, 48};
    const vector<UInt32> expected1 = {createdSegments[0],
                                      createdSegments[1],
                                      createdSegments[2],
                                      createdSegments[3],
                                      createdSegments[6]};
    EXPECT_EQ(expected1,
              ssm.filterSegmentsByCell(
                createdSegments.begin(), createdSegments.end(),
                subset1.begin(), subset1.end()));

    // Subset, some cells without segments
    const vector<UInt32> subset2 = {43, 44, 45, 48};
    const vector<UInt32> expected2 = {createdSegments[3],
                                      createdSegments[6]};
    EXPECT_EQ(expected2,
              ssm.filterSegmentsByCell(
                createdSegments.begin(), createdSegments.end(),
                subset2.begin(), subset2.end()));
  }

  TEST(SegmentMatrixAdapterTest, mapSegmentsToCells)
  {
    SegmentMatrixAdapter<SparseMatrix<>> ssm(2048, 1000);

    const vector<UInt32> cellsWithSegments =
      {42, 42, 42, 43, 44, 45};

    vector<UInt32> createdSegments(cellsWithSegments.size());
    ssm.createSegments(cellsWithSegments.begin(),
                       cellsWithSegments.end(),
                       createdSegments.begin());

    // Map everything
    vector<UInt32> cells1(createdSegments.size());
    ssm.mapSegmentsToCells(createdSegments.begin(),
                           createdSegments.end(),
                           cells1.begin());
    EXPECT_EQ(cellsWithSegments, cells1);

    // Map subset, including duplicates
    vector<UInt32> segmentSubset = {createdSegments[3],
                                    createdSegments[3],
                                    createdSegments[0]};
    vector<UInt32> expectedCells2 = {43, 43, 42};
    vector<UInt32> cells2(segmentSubset.size());
    ssm.mapSegmentsToCells(segmentSubset.begin(),
                           segmentSubset.end(),
                           cells2.begin());
    EXPECT_EQ(expectedCells2, cells2);
  }
}
