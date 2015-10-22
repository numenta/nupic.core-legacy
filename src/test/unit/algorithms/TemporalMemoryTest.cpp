/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

/** @file
 * Implementation of unit tests for TemporalMemory
 */

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

#include "TemporalMemoryTest.hpp"

using namespace std;

namespace nupic {

  void TemporalMemoryTest::setup(TemporalMemory& tm, UInt numColumns)
  {
    vector<UInt> columnDim;
    columnDim.push_back(numColumns);
    tm.initialize(columnDim);
  }

  void TemporalMemoryTest::RunTests()
  {
    setup(tm, 2048);

    testInitInvalidParams();
    testActivateCorrectlyPredictiveCells();
    testActivateCorrectlyPredictiveCellsEmpty();
    testActivateCorrectlyPredictiveCellsOrphan();
    testBurstColumns();
    testBurstColumnsEmpty();
    testLearnOnSegments();
    testComputePredictiveCells();
    testBestMatchingCell();
    testBestMatchingCellFewestSegments();
    testBestMatchingSegment();
    testLeastUsedCell();
    testAdaptSegment();
    testAdaptSegmentToMax();
    testAdaptSegmentToMin();
    testPickCellsToLearnOn();
    testPickCellsToLearnOnAvoidDuplicates();
    testColumnForCell1D();
    testColumnForCell2D();
    testColumnForCellInvalidCell();
    testCellsForColumn1D();
    testCellsForColumn2D();
    testCellsForColumnInvalidColumn();
    testNumberOfColumns();
    testNumberOfCells();
    testMapCellsToColumns();
    testWrite();
    testSaveLoad();

  }

  void TemporalMemoryTest::check_spatial_eq(const TemporalMemory& tm1, const TemporalMemory& tm2)
  {
    NTA_CHECK(tm1.numberOfColumns() == tm2.numberOfColumns());
    NTA_CHECK(tm1.getCellsPerColumn() == tm2.getCellsPerColumn());
    NTA_CHECK(tm1.getActivationThreshold() == tm2.getActivationThreshold());
    NTA_CHECK(tm1.getMinThreshold() == tm2.getMinThreshold());
    NTA_CHECK(tm1.getMaxNewSynapseCount() == tm2.getMaxNewSynapseCount());
    TEST(nupic::nearlyEqual(tm1.getInitialPermanence(), tm2.getInitialPermanence()));
    TEST(nupic::nearlyEqual(tm1.getConnectedPermanence(), tm2.getConnectedPermanence()));
    TEST(nupic::nearlyEqual(tm1.getPermanenceIncrement(), tm2.getPermanenceIncrement()));
    TEST(nupic::nearlyEqual(tm1.getPermanenceDecrement(), tm2.getPermanenceDecrement()));
  }

  void TemporalMemoryTest::testInitInvalidParams()
  {
    // Invalid columnDimensions
    vector<UInt> columnDim = {};
    TemporalMemory tm1;
    SHOULDFAIL(tm1.initialize(columnDim, 32));

    // Invalid cellsPerColumn
    columnDim.push_back(2048);
    SHOULDFAIL(tm1.initialize(columnDim, 0));
  }

  void TemporalMemoryTest::testActivateCorrectlyPredictiveCells()
  {
    set<Cell> prevPredictiveCells = { Cell(0), Cell(237), Cell(1026), Cell(26337), Cell(26339), Cell(55536) };
    set<Cell> prevMatchingCells;
    set<UInt> activeColumns = { 32, 47, 823 };

    set<Cell> activeCells;
    set<Cell> winnerCells;
    set<UInt> predictedColumns;
    set<Cell> predictedInactiveCells;

    tie(activeCells, winnerCells, predictedColumns, predictedInactiveCells) =
      tm.activateCorrectlyPredictiveCells(
        prevPredictiveCells, prevMatchingCells, activeColumns);

    set<Cell> expectedCells = { Cell(1026), Cell(26337), Cell(26339) };
    set<UInt> expectedCols = { 32, 823 };
    set<Cell> expectedInactiveCells;
    NTA_CHECK(check_set_eq(activeCells, expectedCells));
    NTA_CHECK(check_set_eq(winnerCells, expectedCells));
    NTA_CHECK(check_set_eq(predictedColumns, expectedCols));
    NTA_CHECK(check_set_eq(predictedInactiveCells, expectedInactiveCells));
  }

  void TemporalMemoryTest::testActivateCorrectlyPredictiveCellsEmpty()
  {
    {
      set<Cell> prevPredictiveCells;
      set<Cell> prevMatchingCells;
      set<UInt> activeColumns;

      set<Cell> activeCells;
      set<Cell> winnerCells;
      set<UInt> predictedColumns;
      set<Cell> predictedInactiveCells;

      tie(activeCells, winnerCells, predictedColumns, predictedInactiveCells) =
        tm.activateCorrectlyPredictiveCells(
          prevPredictiveCells, prevMatchingCells, activeColumns);

      set<Cell> expectedCells;
      set<UInt> expectedCols;
      set<Cell> expectedInactiveCells;
      NTA_CHECK(check_set_eq(activeCells, expectedCells));
      NTA_CHECK(check_set_eq(winnerCells, expectedCells));
      NTA_CHECK(check_set_eq(predictedColumns, expectedCols));
      NTA_CHECK(check_set_eq(predictedInactiveCells, expectedInactiveCells));
    }

    // No previous predictive cells, with active columns

    {
      set<Cell> prevPredictiveCells;
      set<Cell> prevMatchingCells;
      set<UInt> activeColumns = { 32, 47, 823 };

      set<Cell> activeCells;
      set<Cell> winnerCells;
      set<UInt> predictedColumns;
      set<Cell> predictedInactiveCells;

      tie(activeCells, winnerCells, predictedColumns, predictedInactiveCells) =
        tm.activateCorrectlyPredictiveCells(
          prevPredictiveCells, prevMatchingCells, activeColumns);

      set<Cell> expectedCells;
      set<UInt> expectedCols;
      set<Cell> expectedInactiveCells;
      NTA_CHECK(check_set_eq(activeCells, expectedCells));
      NTA_CHECK(check_set_eq(winnerCells, expectedCells));
      NTA_CHECK(check_set_eq(predictedColumns, expectedCols));
      NTA_CHECK(check_set_eq(predictedInactiveCells, expectedInactiveCells));
    }

    // No active columns, with previously predictive cells

    {
      set<Cell> prevPredictiveCells = { Cell(0), Cell(237), Cell(1026), Cell(26337), Cell(26339), Cell(55536) };
      set<Cell> prevMatchingCells;
      set<UInt> activeColumns;

      set<Cell> activeCells;
      set<Cell> winnerCells;
      set<UInt> predictedColumns;
      set<Cell> predictedInactiveCells;

      tie(activeCells, winnerCells, predictedColumns, predictedInactiveCells) =
        tm.activateCorrectlyPredictiveCells(
          prevPredictiveCells, prevMatchingCells, activeColumns);

      set<Cell> expectedCells;
      set<UInt> expectedCols;
      set<Cell> expectedInactiveCells;
      NTA_CHECK(check_set_eq(activeCells, expectedCells));
      NTA_CHECK(check_set_eq(winnerCells, expectedCells));
      NTA_CHECK(check_set_eq(predictedColumns, expectedCols));
      NTA_CHECK(check_set_eq(predictedInactiveCells, expectedInactiveCells));
    }
  }

  void TemporalMemoryTest::testActivateCorrectlyPredictiveCellsOrphan()
  {
    TemporalMemory tm;
    tm.initialize();
    tm.setPredictedSegmentDecrement(0.001);

    set<Cell> prevPredictiveCells;
    set<UInt> activeColumns = { 32, 47, 823 };
    set<Cell> prevMatchingCells = { 32, 47 };

    set<Cell> activeCells;
    set<Cell> winnerCells;
    set<UInt> predictedColumns;
    set<Cell> predictedInactiveCells;

    tie(activeCells, winnerCells, predictedColumns, predictedInactiveCells) =
      tm.activateCorrectlyPredictiveCells(
        prevPredictiveCells,
        prevMatchingCells,
        activeColumns);

    set<Cell> expectedCells;
    set<UInt> expectedCols;
    set<Cell> expectedInactiveCells = { 32, 47 };
    NTA_CHECK(check_set_eq(activeCells, expectedCells));
    NTA_CHECK(check_set_eq(winnerCells, expectedCells));
    NTA_CHECK(check_set_eq(predictedColumns, expectedCols));
    NTA_CHECK(check_set_eq(predictedInactiveCells, expectedInactiveCells));
  }

  void TemporalMemoryTest::testBurstColumns()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 4);
    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(49), 0.9);
    connections.createSynapse(segment, Cell(3), 0.8);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(733), 0.7);

    segment = connections.createSegment(Cell(108));
    connections.createSynapse(segment, Cell(486), 0.9);

    set<UInt> activeColumns = { 0, 1, 26 };
    set<UInt> predictiveCols = { 26 };
    set<Cell> prevActiveCells = { Cell(23), Cell(37), Cell(49), Cell(733) };
    set<Cell> prevWinnerCells = { Cell(23), Cell(37), Cell(49), Cell(733) };

    set<Cell> activeCells;
    set<Cell> winnerCells;
    vector<Segment> learningSegments;

    tie(activeCells, winnerCells, learningSegments) =
      tm.burstColumns(activeColumns, predictiveCols, prevActiveCells, prevWinnerCells, connections);

    set<Cell> expectedActiveCells = { Cell(0), Cell(1), Cell(2), Cell(3), Cell(4), Cell(5), Cell(6), Cell(7) };
    set<Cell> expectedWinnerCells = { Cell(0), Cell(4) }; // 4 is randomly chosen cell
    vector<Segment> expectedLearningSegments = { Segment(0, Cell(0)), Segment(0, Cell(4)) };
    NTA_CHECK(check_set_eq(activeCells, expectedActiveCells));
    NTA_CHECK(check_set_eq(winnerCells, expectedWinnerCells));
    NTA_CHECK(check_vector_eq(learningSegments, expectedLearningSegments));

    // Check that new segment was added to winner cell(4) in column 1
    vector<Segment> segments = connections.segmentsForCell(4);
    vector<Segment> expectedSegments = { Segment(0, Cell(4)) };
    NTA_CHECK(check_vector_eq(segments, expectedSegments));
  }

  void TemporalMemoryTest::testBurstColumnsEmpty()
  {
    set<UInt> activeColumns;
    set<UInt> predictiveCols;
    set<Cell> prevActiveCells;
    set<Cell> prevWinnerCells;
    Connections connections = tm.connections;

    set<Cell> activeCells;
    set<Cell> winnerCells;
    vector<Segment> learningSegments;

    tie(activeCells, winnerCells, learningSegments) =
      tm.burstColumns(activeColumns, predictiveCols, prevActiveCells, prevWinnerCells, connections);

    set<Cell> expectedActiveCells;
    set<Cell> expectedWinnerCells;
    vector<Segment> expectedLearningSegments;
    NTA_CHECK(check_set_eq(activeCells, expectedActiveCells));
    NTA_CHECK(check_set_eq(winnerCells, expectedWinnerCells));
    NTA_CHECK(check_vector_eq(learningSegments, expectedLearningSegments));
  }

  void TemporalMemoryTest::testLearnOnSegments()
  {
    vector<Synapse> synapses;
    bool eq;

    TemporalMemory tm;
    setup(tm, 2048);
    tm.setMaxNewSynapseCount(2);

    Connections connections = tm.connections;
    Segment segment0 = connections.createSegment(Cell(0));
    connections.createSynapse(segment0, Cell(23), 0.6);
    connections.createSynapse(segment0, Cell(37), 0.4);
    connections.createSynapse(segment0, Cell(477), 0.9);

    Segment segment1 = connections.createSegment(Cell(1));
    connections.createSynapse(segment1, Cell(733), 0.7);

    Segment segment2 = connections.createSegment(Cell(8));
    connections.createSynapse(segment2, Cell(486), 0.9);

    Segment segment3 = connections.createSegment(Cell(100));

    vector<Segment> prevActiveSegments = { segment0, segment2 };
    vector<Segment> learningSegments = { segment1, segment3 };
    set<Cell> prevActiveCells = { Cell(23), Cell(37), Cell(733) };
    set<Cell> winnerCells = { Cell(0) };
    set<Cell> prevWinnerCells = { Cell(10), Cell(11), Cell(12), Cell(13), Cell(14) };
    vector<Segment> prevMatchingSegments;
    set<Cell> predictedInactiveCells;

    tm.learnOnSegments(
      prevActiveSegments,
      learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      connections,
      predictedInactiveCells,
      prevMatchingSegments);

    // Check segment 0
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment0)).permanence, Permanence(0.7));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(1, segment0)).permanence, Permanence(0.5));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(2, segment0)).permanence, Permanence(0.8));
    EXPECT_TRUE(eq);

    // Check segment 1
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment1)).permanence, Permanence(0.8));
    EXPECT_TRUE(eq);
    synapses = connections.synapsesForSegment(segment1);
    ASSERT_EQ(synapses.size(), 2);

    // Check segment 2
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment2)).permanence, Permanence(0.9));
    EXPECT_TRUE(eq);
    synapses = connections.synapsesForSegment(segment2);
    ASSERT_EQ(synapses.size(), 1);

    // Check segment 3
    synapses = connections.synapsesForSegment(segment3);
    ASSERT_EQ(synapses.size(), 2);
  }

  void TemporalMemoryTest::testComputePredictiveCells()
  {
    TemporalMemory tm;
    setup(tm, 2048);
    tm.setActivationThreshold(2);
    tm.setMinThreshold(2);
    tm.setPredictedSegmentDecrement(0.004);

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.5);
    connections.createSynapse(segment, Cell(477), 0.9);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(733), 0.7);
    connections.createSynapse(segment, Cell(733), 0.4);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(974), 0.9);

    segment = connections.createSegment(Cell(8));
    connections.createSynapse(segment, Cell(486), 0.9);

    segment = connections.createSegment(Cell(100));

    set<Cell> activeCells = { Cell(23), Cell(37), Cell(733), Cell(974) };

    vector<Segment> activeSegments;
    set<Cell> predictiveCells;

    vector<Segment> matchingSegments;
    set<Cell> matchingCells;

    tie(activeSegments, predictiveCells, matchingSegments, matchingCells) =
      tm.computePredictiveCells(activeCells, connections);

    vector<Segment> expectedActiveSegments = { Segment(0, Cell(0)) };
    set<Cell> expectedPredictiveCells = { Cell(0) };
    vector<Segment> expectedMatchingSegments = { Segment(0, Cell(0)), Segment(0, Cell(1)) };
    set<Cell> expectedMatchingCells = { Cell(0), Cell(1) };
    NTA_CHECK(check_vector_eq(activeSegments, expectedActiveSegments));
    NTA_CHECK(check_set_eq(predictiveCells, expectedPredictiveCells));
    NTA_CHECK(check_vector_eq(matchingSegments, expectedMatchingSegments));
    NTA_CHECK(check_set_eq(matchingCells, expectedMatchingCells));
  }

  void TemporalMemoryTest::testBestMatchingCell()
  {
    bool foundCell, foundSegment;
    Cell bestCell;
    Segment bestSegment;

    TemporalMemory tm;
    setup(tm, 2048);
    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);
    tm.seed_(42);

    Connections connections = tm.connections;

    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(49), 0.9);
    connections.createSynapse(segment, Cell(3), 0.8);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(733), 0.7);

    segment = connections.createSegment(Cell(108));
    connections.createSynapse(segment, Cell(486), 0.9);

    set<Cell> activeCells = { Cell(23), Cell(37), Cell(49), Cell(733) };
    vector<Cell> cellsForColumn = tm.cellsForColumn(0);

    tie(foundCell, bestCell, foundSegment, bestSegment) =
      tm.bestMatchingCell(cellsForColumn, activeCells, connections);

    ASSERT_EQ(bestCell, Cell(0));
    ASSERT_EQ(bestSegment, Segment(0, Cell(0)));

    cellsForColumn = tm.cellsForColumn(3);
    tie(foundCell, bestCell, foundSegment, bestSegment) =
      tm.bestMatchingCell(cellsForColumn, activeCells, connections);
    ASSERT_EQ(bestCell, Cell(103)); // Random cell from column

    cellsForColumn = tm.cellsForColumn(999);
    tie(foundCell, bestCell, foundSegment, bestSegment) =
      tm.bestMatchingCell(cellsForColumn, activeCells, connections);

    ASSERT_EQ(bestCell, Cell(31979)); // Random cell from column
  }

  void TemporalMemoryTest::testBestMatchingCellFewestSegments()
  {
    bool foundCell, foundSegment;
    Cell cell;
    Segment segment;

    TemporalMemory tm;
    tm.initialize(vector<UInt>{2}, 2);
    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);
    tm.seed_(42);

    Connections connections = tm.connections;
    connections.createSynapse(connections.createSegment(Cell(0)), 3, 0.3);

    set<Cell> activeSynapsesForSegment;

    for (int i = 0; i < 100; i++)
    {
      // Never pick cell 0, always pick cell 1
      vector<Cell> cellsForColumn = tm.cellsForColumn(0);
      tie(foundCell, cell, foundSegment, segment) =
        tm.bestMatchingCell(cellsForColumn, activeSynapsesForSegment, connections);
      ASSERT_EQ(cell, Cell(1));
    }
  }

  void TemporalMemoryTest::testBestMatchingSegment()
  {
    Int numActiveSynapses;
    Segment bestSegment;
    bool found;

    TemporalMemory tm;
    setup(tm, 2048);
    tm.setMinThreshold(1);

    Connections connections = tm.connections;

    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(49), 0.9);
    connections.createSynapse(segment, Cell(3), 0.8);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(733), 0.7);

    segment = connections.createSegment(Cell(8));
    connections.createSynapse(segment, Cell(486), 0.9);

    set<Cell> activeCells = { Cell(23), Cell(37), Cell(49), Cell(733) };

    Cell cell;
    cell.idx = 0;

    tie(found, bestSegment, numActiveSynapses) = tm.bestMatchingSegment(cell, activeCells, connections);
    if (found) ASSERT_EQ(bestSegment, Segment(0, Cell(0)));
    ASSERT_EQ(numActiveSynapses, 2);

    cell.idx = 1;
    tie(found, bestSegment, numActiveSynapses) = tm.bestMatchingSegment(cell, activeCells, connections);
    if (found) ASSERT_EQ(bestSegment, Segment(0, Cell(1)));
    ASSERT_EQ(numActiveSynapses, 1);

    cell.idx = 8;
    tie(found, bestSegment, numActiveSynapses) = tm.bestMatchingSegment(cell, activeCells, connections);
    ASSERT_EQ(found, false);

    cell.idx = 100;
    tie(found, bestSegment, numActiveSynapses) = tm.bestMatchingSegment(cell, activeCells, connections);
    ASSERT_EQ(found, false);
  }

  void TemporalMemoryTest::testLeastUsedCell()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2}, 2);
    tm.seed_(42);

    Connections connections = tm.connections;
    connections.createSynapse(connections.createSegment(Cell(0)), 3, 0.3);

    set<Cell> cells;
    Segment segment;
    Cell cell;
    bool foundCell, foundSegment;

    for (int i = 0; i < 100; i++)
    {
      // Never pick cell 0, always pick cell 1
      vector<Cell> cellsForColumn = tm.cellsForColumn(0);
      tie(foundCell, cell, foundSegment, segment) =
        tm.bestMatchingCell(cellsForColumn, cells, connections);
      ASSERT_EQ(cell, Cell(1));
    }
  }

  void TemporalMemoryTest::testAdaptSegment()
  {
    vector<Synapse> synapses;
    bool eq;

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    synapses = vector<Synapse>{ Synapse(0, segment), Synapse(1, segment) };

    tm.adaptSegment(
      segment, synapses, connections,  tm.getPermanenceIncrement(), tm.getPermanenceDecrement());

    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Permanence(0.7));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(1, segment)).permanence, Permanence(0.5));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(2, segment)).permanence, Permanence(0.8));
    EXPECT_TRUE(eq);
  }

  void TemporalMemoryTest::testAdaptSegmentToMax()
  {
    vector<Synapse> synapses;
    bool eq;

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    synapses.push_back(connections.createSynapse(segment, Cell(23), 0.9));

    tm.adaptSegment(
      segment, synapses, connections, tm.getPermanenceIncrement(), tm.getPermanenceDecrement());

    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Permanence(1.0));
    EXPECT_TRUE(eq);

    // Now permanence should be at min
    tm.adaptSegment(
      segment, synapses, connections, tm.getPermanenceIncrement(), tm.getPermanenceDecrement());

    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Permanence(1.0));
    EXPECT_TRUE(eq);
  }

  void TemporalMemoryTest::testAdaptSegmentToMin()
  {
    vector<Synapse> synapses;

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.1);

    tm.adaptSegment(
      segment, synapses, connections, tm.getPermanenceIncrement(), tm.getPermanenceDecrement());

    synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 0);
  }

  void TemporalMemoryTest::testPickCellsToLearnOn()
  {
    TemporalMemory tm;
    setup(tm, 2048);
    tm.seed_(42);

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));

    set<Cell> winnerCells = { Cell(4), Cell(47), Cell(58), Cell(93) };
    set<Cell> learningCells, expectedCells;

    expectedCells = set<Cell>{ Cell(4), Cell(93) }; // Randomly picked
    learningCells = tm.pickCellsToLearnOn(2, segment, winnerCells, connections);
    NTA_CHECK(check_set_eq(learningCells, expectedCells));

    expectedCells = set<Cell>{ Cell(4), Cell(47), Cell(58), Cell(93) };
    learningCells = tm.pickCellsToLearnOn(100, segment, winnerCells, connections);
    NTA_CHECK(check_set_eq(learningCells, expectedCells));

    expectedCells = set<Cell>{};
    learningCells = tm.pickCellsToLearnOn(0, segment, winnerCells, connections);
    NTA_CHECK(check_set_eq(learningCells, expectedCells));
  }

  void TemporalMemoryTest::testPickCellsToLearnOnAvoidDuplicates()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    Connections connections = tm.connections;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, 23, 0.6);

    set<Cell> winnerCells = { Cell(23) };

    // Ensure that no additional(duplicate) cells were picked
    set<Cell> expectedCells;
    set<Cell> learningCells = tm.pickCellsToLearnOn(2, segment, winnerCells, connections);
    NTA_CHECK(check_set_eq(learningCells, expectedCells));
  }

  void TemporalMemoryTest::testColumnForCell1D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    Cell cell;

    cell.idx = 0;
    TEST(tm.columnForCell(cell) == 0);
    cell.idx = 4;
    TEST(tm.columnForCell(cell) == 0);
    cell.idx = 5;
    TEST(tm.columnForCell(cell) == 1);
    cell.idx = 10239;
    TEST(tm.columnForCell(cell) == 2047);
  }

  void TemporalMemoryTest::testColumnForCell2D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    Cell cell;

    cell.idx = 0;
    TEST(tm.columnForCell(cell) == 0);
    cell.idx = 3;
    TEST(tm.columnForCell(cell) == 0);
    cell.idx = 4;
    TEST(tm.columnForCell(cell) == 1);
    cell.idx = 16383;
    TEST(tm.columnForCell(cell) == 4095);
  }

  void TemporalMemoryTest::testColumnForCellInvalidCell()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    Cell cell;

    cell.idx = 16383;
    EXPECT_NO_THROW(tm.columnForCell(cell));
    cell.idx = 16384;
    EXPECT_THROW(tm.columnForCell(cell), std::exception);
    cell.idx = -1;
    EXPECT_THROW(tm.columnForCell(cell), std::exception);
  }

  void TemporalMemoryTest::testCellsForColumn1D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    vector<Cell> expectedCells = { Cell(5), Cell(6), Cell(7), Cell(8), Cell(9) };
    vector<Cell> cellsForColumn = tm.cellsForColumn(1);
    NTA_CHECK(check_vector_eq(cellsForColumn, expectedCells));
  }

  void TemporalMemoryTest::testCellsForColumn2D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    vector<Cell> expectedCells = { Cell(256), Cell(257), Cell(258), Cell(259) };
    vector<Cell> cellsForColumn = tm.cellsForColumn(64);
    NTA_CHECK(check_vector_eq(cellsForColumn, expectedCells));
  }

  void TemporalMemoryTest::testCellsForColumnInvalidColumn()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    EXPECT_NO_THROW(tm.cellsForColumn(4095));
    EXPECT_THROW(tm.cellsForColumn(4096), std::exception);
    EXPECT_THROW(tm.cellsForColumn(-1), std::exception);
  }

  void TemporalMemoryTest::testNumberOfColumns()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    int numOfColumns = tm.numberOfColumns();
    ASSERT_EQ(numOfColumns, 64 * 64);
  }

  void TemporalMemoryTest::testNumberOfCells()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    Int numberOfCells = tm.numberOfCells();
    ASSERT_EQ(numberOfCells, 64 * 64 * 32);
  }

  void TemporalMemoryTest::testMapCellsToColumns()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{100}, 4);

    set<Cell> cells = { Cell(0), Cell(1), Cell(2), Cell(5), Cell(399) };
    map<Int, set<Cell>> columnsForCells = tm.mapCellsToColumns(cells);

    set<Cell> expectedCells = { Cell(0), Cell(1), Cell(2) };
    NTA_CHECK(check_set_eq(columnsForCells[0], expectedCells));
    expectedCells = { Cell(5) };
    NTA_CHECK(check_set_eq(columnsForCells[1], expectedCells));
    expectedCells = { Cell(399) };
    NTA_CHECK(check_set_eq(columnsForCells[99], expectedCells));
  }

  void TemporalMemoryTest::testSaveLoad()
  {
    const char* filename = "TemporalMemorySerialization.tmp";
    TemporalMemory tm1, tm2;
    vector<UInt> columnDim;
    UInt numColumns = 12;

    columnDim.push_back(numColumns);
    tm1.initialize(columnDim);

    ofstream outfile;
    outfile.open(filename, ios::binary);
    tm1.save(outfile);
    outfile.close();

    ifstream infile(filename, ios::binary);
    tm2.load(infile);
    infile.close();

    check_spatial_eq(tm1, tm2);

    int ret = ::remove(filename);
    NTA_CHECK(ret == 0) << "Failed to delete " << filename;
  }

  void TemporalMemoryTest::testWrite()
  {
    TemporalMemory tm1, tm2;

    tm1.initialize({ 100 }, 4, 7, 0.37, 0.58, 4, 18, 0.23, 0.08, 0.0, 91);

    // Run some data through before serializing
    /*
    PatternMachine patternMachine = PatternMachine(100, 4);
    SequenceMachine sequenceMachine = SequenceMachine(self.patternMachine);
    Sequence sequence = self.sequenceMachine.generateFromNumbers(range(5));
    */
    vector<vector<UInt>> sequence = 
    { 
      { 83, 53, 70, 45 },
      { 8, 65, 67, 59 },
      { 25, 98, 99, 39 },
      { 66, 11, 78, 14 },
      { 96, 87, 69, 95 } };

    for (UInt i = 0; i < 3; i++)
    {
      for (vector<UInt> pattern : sequence)
        tm1.compute(pattern.size(), pattern.data());
    }

    // Write and read back the proto
    stringstream ss;
    tm1.write(ss);
    tm2.read(ss);

    // Check that the two temporal memory objects have the same attributes
    check_spatial_eq(tm1, tm2);

    tm1.compute(sequence[0].size(), sequence[0].data());
    tm2.compute(sequence[0].size(), sequence[0].data());
    ASSERT_EQ(tm1.activeCells, tm2.activeCells);
    ASSERT_EQ(tm1.predictiveCells, tm2.predictiveCells);
    ASSERT_EQ(tm1.winnerCells, tm2.winnerCells);
    ASSERT_EQ(tm1.connections, tm2.connections);

    tm1.compute(sequence[3].size(), sequence[3].data());
    tm2.compute(sequence[3].size(), sequence[3].data());
    ASSERT_EQ(tm1.activeCells, tm2.activeCells);
    ASSERT_EQ(tm1.predictiveCells, tm2.predictiveCells);
    ASSERT_EQ(tm1.winnerCells, tm2.winnerCells);
    ASSERT_EQ(tm1.connections, tm2.connections);
  }

  void TemporalMemoryTest::print_vec(UInt arr[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      cout << arr[i] << " ";
    }
    cout << endl;
  }

  void TemporalMemoryTest::print_vec(Real arr[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      cout << arr[i] << " ";
    }
    cout << endl;
  }

  void TemporalMemoryTest::print_vec(vector<UInt>& vec)
  {
    for (auto & elem : vec) {
      cout << elem << " ";
    }
    cout << endl;
  }

  void TemporalMemoryTest::print_vec(vector<Real>& vec)
  {
    for (auto & elem : vec) {
      cout << elem << " ";
    }
    cout << endl;
  }

  bool TemporalMemoryTest::almost_eq(Real a, Real b)
  {
    Real diff = a - b;
    return (diff > -1e-5 && diff < 1e-5);
  }

  bool TemporalMemoryTest::check_vector_eq(UInt arr[], vector<UInt>& vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (arr[i] != vec[i]) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(Real arr[], vector<Real>& vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (!almost_eq(arr[i], vec[i])) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(UInt arr1[], UInt arr2[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      if (arr1[i] != arr2[i]) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(Real arr1[], Real arr2[], UInt n)
  {
    for (UInt i = 0; i < n; i++) {
      if (!almost_eq(arr1[i], arr2[i])) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_set_eq(set<UInt>& vec1, set<UInt>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (UInt i : vec2) {
      if (vec1.find(i) == vec1.end()) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_set_eq(set<Cell>& vec1, set<Cell>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (Cell cell : vec2) {
      if (vec1.find(cell) == vec1.end()) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(vector<UInt>& vec1, vector<UInt>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (UInt i = 0; i < vec1.size(); i++) {
      if (vec1[i] != vec2[i]) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(vector<Cell>& vec1, vector<Cell>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (UInt i = 0; i < vec1.size(); i++) {
      if (vec1[i].idx != vec2[i].idx) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(vector<Segment>& vec1, vector<Segment>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (UInt i = 0; i < vec1.size(); i++) {
      if (vec1[i].idx != vec2[i].idx || vec1[i].cell.idx != vec2[i].cell.idx) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(vector<Segment*>& vec1, vector<Segment>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (UInt i = 0; i < vec1.size(); i++) {
      if (vec1[i]->idx != vec2[i].idx || vec1[i]->cell.idx != vec2[i].cell.idx) {
        return false;
      }
    }
    return true;
  }
} // end namespace nupic
