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

#include <nupic/algorithms/TemporalMemory.hpp>
#include "gtest/gtest.h"

using namespace nupic::algorithms::temporal_memory;
using namespace std;

#define EPSILON 0.0000001

namespace {
  void check_tm_eq(const TemporalMemory& tm1, const TemporalMemory& tm2)
  {
    ASSERT_EQ(tm1.numberOfColumns(), tm2.numberOfColumns());
    ASSERT_EQ(tm1.getCellsPerColumn(), tm2.getCellsPerColumn());
    ASSERT_EQ(tm1.getActivationThreshold(), tm2.getActivationThreshold());
    ASSERT_EQ(tm1.getMinThreshold(), tm2.getMinThreshold());
    ASSERT_EQ(tm1.getMaxNewSynapseCount(), tm2.getMaxNewSynapseCount());
    ASSERT_NEAR(tm1.getInitialPermanence(), tm2.getInitialPermanence(), EPSILON);
    ASSERT_NEAR(tm1.getConnectedPermanence(), tm2.getConnectedPermanence(), EPSILON);
    ASSERT_NEAR(tm1.getPermanenceIncrement(), tm2.getPermanenceIncrement(), EPSILON);
    ASSERT_NEAR(tm1.getPermanenceDecrement(), tm2.getPermanenceDecrement(), EPSILON);
  }

  vector<Cell> cellsForColumnCell(TemporalMemory& tm, UInt32 column)
  {
    vector<Cell> cellsInColumn;
    for (CellIdx cell : tm.cellsForColumn(column))
    {
      cellsInColumn.push_back(Cell(cell));
    }

    return cellsInColumn;
  }

  TEST(TemporalMemoryTest, testInitInvalidParams)
  {
    // Invalid columnDimensions
    vector<UInt> columnDim = {};
    TemporalMemory tm1;
    EXPECT_THROW(tm1.initialize(columnDim, 32), exception);

    // Invalid cellsPerColumn
    columnDim.push_back(2048);
    EXPECT_THROW(tm1.initialize(columnDim, 0), exception);
  }

  TEST(TemporalMemoryTest, ActivateCorrectlyPredictiveCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 4;
    const UInt columns_a[4] = {0, 8, 16, 24};
    const UInt columns_b[4] = {1, 9, 17, 25};
    const vector<CellIdx> winners_a = {0, 33, 66, 98};
    const vector<CellIdx> predicted_b = {4, 37, 70, 102};

    for (CellIdx postsynaptic : predicted_b)
    {
      Segment segment = tm.connections.createSegment(Cell(postsynaptic));
      for (CellIdx presynaptic : winners_a)
      {
        tm.connections.createSynapse(segment, Cell(presynaptic), 0.5);
      }
    }

    tm.compute(numActiveColumns, columns_a, false);

    ASSERT_EQ(predicted_b, tm.getPredictiveCells());

    tm.compute(numActiveColumns, columns_b, false);

    EXPECT_EQ(predicted_b, tm.getActiveCells());
  }

  TEST(TemporalMemoryTest, BurstUnpredictedColumns)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 4;
    const UInt columns_a[4] = {0, 8, 16, 24};
    const vector<CellIdx> bursting_a = {0, 1, 2, 3,
                                        32, 33, 34, 35,
                                        64, 65, 66, 67,
                                        96, 97, 98, 99};

    tm.compute(numActiveColumns, columns_a, false);

    EXPECT_EQ(bursting_a, tm.getActiveCells());
  }

  TEST(TemporalMemoryTest, BurstingAndNonBursting)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 4;
    const UInt columns_a[4] = {0, 8, 16, 24};
    const UInt columns_actual_b[4] = {1, 9, 20, 30};
    const vector<CellIdx> winners_a = {0, 33, 66, 98};
    const vector<CellIdx> predicted_b = {4, 37, 70, 102}; // columns {1, 9, 17, 25}
    const vector<CellIdx> actual_b = {4,
                                      37,
                                      80, 81, 82, 83,
                                      120, 121, 122, 123};

    for (CellIdx postsynaptic : predicted_b)
    {
      Segment segment = tm.connections.createSegment(Cell(postsynaptic));
      for (CellIdx presynaptic : winners_a)
      {
        tm.connections.createSynapse(segment, Cell(presynaptic), 0.5);
      }
    }

    tm.compute(numActiveColumns, columns_a, false);

    ASSERT_EQ(predicted_b, tm.getPredictiveCells());

    tm.compute(numActiveColumns, columns_actual_b, false);

    EXPECT_EQ(actual_b, tm.getActiveCells());
  }

  TEST(TemporalMemoryTest, ZeroActiveColumns)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.02,
      /*seed*/ 42
      );

    const UInt zeroColumns[0] = {};
    const vector<CellIdx> zeroCells = {};

    tm.compute(0, zeroColumns, true);

    EXPECT_EQ(zeroCells, tm.getActiveCells());
    EXPECT_EQ(zeroCells, tm.getPredictiveCells());

    // Now make some cells predictive.

    const UInt numActiveColumns = 4;
    const UInt columns_a[4] = {0, 8, 16, 24};
    const vector<CellIdx> winners_a = {0, 33, 66, 98};
    const vector<CellIdx> predicted_b = {4, 37, 70, 102};

    for (CellIdx postsynaptic : predicted_b)
    {
      Segment segment = tm.connections.createSegment(Cell(postsynaptic));
      for (CellIdx presynaptic : winners_a)
      {
        tm.connections.createSynapse(segment, Cell(presynaptic), 0.5);
      }
    }

    tm.compute(numActiveColumns, columns_a, false);

    ASSERT_EQ(predicted_b, tm.getPredictiveCells());

    tm.compute(0, zeroColumns, true);

    EXPECT_EQ(zeroCells, tm.getActiveCells());
    EXPECT_EQ(zeroCells, tm.getPredictiveCells());
  }

  vector<Permanence> getPermanences(TemporalMemory& tm, Segment segment)
  {
    vector<Permanence> perms;
    for (Synapse synapse : tm.connections.synapsesForSegment(segment))
    {
      perms.push_back(tm.connections.dataForSynapse(synapse).permanence);
    }
    return perms;
  }

  TEST(TemporalMemoryTest, ReinforceCorrectlyActiveSegments)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.02,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<Cell> previousActiveCells = {{0}, {1}, {2}, {3}};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> activeCells = {5};
    const Cell activeCell = {5};

    Segment activeSegment1 = tm.connections.createSegment(activeCell);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[2], 0.5);
    tm.connections.createSynapse(activeSegment1, Cell(81), 0.5);
    tm.connections.createSynapse(activeSegment1, Cell(82), 0.01);

    // Verifies:
    // - Active synapses reinforced
    // - Inactive synapses punished
    //   - 1 destroyed
    // - No growth happens, despite the fact that there's another previous
    //   active cell available
    const vector<Permanence> activeSegment1_result = {0.60, 0.60, 0.60, 0.40};

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    ASSERT_EQ(activeCells, tm.getActiveCells());

    EXPECT_EQ(activeSegment1_result, getPermanences(tm, activeSegment1));
  }

  TEST(TemporalMemoryTest, SometimesReinforceCorrectlyMatchingSegments)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.02,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 2;
    const UInt previousActiveColumns[2] = {0, 4};
    const vector<Cell> previousActiveCells = {{0}, {1}, {2}, {3},
                                              {16}, {17}, {18}, {19}};
    const UInt activeColumns[2] = {1, 5};
    const vector<CellIdx> activeCells = {5,
                                         20, 21, 22, 23};
    const Cell previousInactiveCell = {81};
    const Cell correctlyPredictedCell = {5};

    // Create 4 segments:
    // - 1 that predicts a column
    // - 1 that matches the same cell
    // - 1 that matches a cell in a bursting column, selected
    // - 1 that matches a cell in a bursting column, not selected

    // Keep one of the columns from bursting
    Segment correctlyActiveSegment =
      tm.connections.createSegment(correctlyPredictedCell);
    for (Cell cell : previousActiveCells)
    {
      tm.connections.createSynapse(correctlyActiveSegment, cell, 0.5);
    }

    // Matching segment on a cell that was predicted by another segment
    Segment matchingSegment1 =
      tm.connections.createSegment(correctlyPredictedCell);
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(matchingSegment1, previousInactiveCell, 0.80);
    const vector<Permanence> matchingSegment1_result = {0.5, 0.5, 0.80};

    // Matching segment in a bursting column: selected
    Segment matchingSegment2 =
      tm.connections.createSegment(Cell(21));
    tm.connections.createSynapse(matchingSegment2, previousActiveCells[0], 0.4);
    tm.connections.createSynapse(matchingSegment2, previousActiveCells[1], 0.4);
    tm.connections.createSynapse(matchingSegment2, previousActiveCells[2], 0.4);
    tm.connections.createSynapse(matchingSegment2, Cell(81), 0.8);
    tm.connections.createSynapse(matchingSegment2, Cell(82), 0.01);

    // Verifies:
    // - Active synapses reinforced
    // - New synapses grown
    // - Inactive synapses punished
    //   - 1 destroyed
    const vector<Permanence> matchingSegment2_result = {0.5, 0.5, 0.5, 0.7, 0.2};

    // Matching segment in a bursting column: not selected
    Segment matchingSegment3 =
      tm.connections.createSegment(Cell(22));
    tm.connections.createSynapse(matchingSegment3, previousActiveCells[0], 0.4);
    tm.connections.createSynapse(matchingSegment3, previousActiveCells[1], 0.4);
    tm.connections.createSynapse(matchingSegment3, Cell(81), 0.8);
    tm.connections.createSynapse(matchingSegment3, Cell(82), 0.01);
    const vector<Permanence> matchingSegment3_result = {0.4, 0.4, 0.8, 0.01};

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    ASSERT_EQ(activeCells, tm.getActiveCells());

    EXPECT_EQ(matchingSegment1_result, getPermanences(tm, matchingSegment1));
    EXPECT_EQ(matchingSegment2_result, getPermanences(tm, matchingSegment2));
    EXPECT_EQ(matchingSegment3_result, getPermanences(tm, matchingSegment3));
  }

  TEST(TemporalMemoryTest, PunishSegmentsInInactiveColumns)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.02,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<Cell> previousActiveCells = {{0}, {1}, {2}, {3}};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> activeCells = {5};
    const Cell previousInactiveCell = {81};
    const Cell activeCell = {5};

    // Predict the active column so that it's not bursting.
    Segment correctlyActiveSegment = tm.connections.createSegment(activeCell);
    for (Cell cell : previousActiveCells)
    {
      tm.connections.createSynapse(correctlyActiveSegment, cell, 0.5);
    }

    // Active segment on inactive cell
    Segment activeSegment1 = tm.connections.createSegment(Cell(42));
    tm.connections.createSynapse(activeSegment1, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[2], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[3], 0.01);
    tm.connections.createSynapse(activeSegment1, previousInactiveCell, 0.80);
    const vector<Permanence> activeSegment1_result = {0.48, 0.48, 0.48, 0.80};

    // Matching segment on an inactive cell in an active column
    Segment matchingSegment1 = tm.connections.createSegment(Cell(6));
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[0], 0.4);
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[1], 0.4);
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[2], 0.4);
    tm.connections.createSynapse(matchingSegment1, previousActiveCells[3], 0.01);
    tm.connections.createSynapse(matchingSegment1, previousInactiveCell, 0.80);
    const vector<Permanence> matchingSegment1_result = {0.4, 0.4, 0.4, 0.01, 0.80};

    // Matching segment on an inactive cell in an inactive column
    Segment matchingSegment2 = tm.connections.createSegment(Cell(50));
    tm.connections.createSynapse(matchingSegment2, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(matchingSegment2, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(matchingSegment2, previousInactiveCell, 0.80);
    const vector<Permanence> matchingSegment2_result = {0.48, 0.48, 0.80};

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    ASSERT_EQ(activeCells, tm.getActiveCells());

    EXPECT_EQ(activeSegment1_result, getPermanences(tm, activeSegment1));
    EXPECT_EQ(matchingSegment1_result, getPermanences(tm, matchingSegment1));
    EXPECT_EQ(matchingSegment2_result, getPermanences(tm, matchingSegment2));
  }

  TEST(TemporalMemoryTest, AddSegmentToCellWithFewestSegments)
  {
    bool grewOnCell1 = false;
    bool grewOnCell2 = false;
    for (UInt seed = 0; seed < 100; seed++)
    {
      TemporalMemory tm(
        /*columnDimensions*/ {32},
        /*cellsPerColumn*/ 4,
        /*activationThreshold*/ 3,
        /*initialPermanence*/ 0.2,
        /*connectedPermanence*/ 0.50,
        /*minThreshold*/ 2,
        /*maxNewSynapseCount*/ 4,
        /*permanenceIncrement*/ 0.10,
        /*permanenceDecrement*/ 0.10,
        /*predictedSegmentDecrement*/ 0.02,
        /*seed*/ seed
        );

      // enough for 4 winner cells
      const UInt previousActiveColumns[4] ={1, 2, 3, 4};
      const UInt activeColumns[1] = {0};
      const vector<Cell> previousActiveCells =
        {{4}, {5}, {6}, {7}}; // (there are more)
      vector<Cell> nonmatchingCells = {{0}, {3}};
      vector<CellIdx> activeCells = {0, 1, 2, 3};

      Segment segment1 = tm.connections.createSegment(nonmatchingCells[0]);
      tm.connections.createSynapse(segment1, previousActiveCells[0], 0.5);
      Segment segment2 = tm.connections.createSegment(nonmatchingCells[1]);
      tm.connections.createSynapse(segment2, previousActiveCells[1], 0.5);

      tm.compute(4, previousActiveColumns, true);
      tm.compute(1, activeColumns, true);

      ASSERT_EQ(activeCells, tm.getActiveCells());

      EXPECT_EQ(3, tm.connections.numSegments());
      EXPECT_EQ(1, tm.connections.segmentsForCell({0}).size());
      EXPECT_EQ(1, tm.connections.segmentsForCell({3}).size());
      EXPECT_EQ(1, tm.connections.numSynapses(segment1));
      EXPECT_EQ(1, tm.connections.numSynapses(segment2));

      Segment grownSegment;
      vector<Segment> segments = tm.connections.segmentsForCell({1});
      if (segments.empty())
      {
        vector<Segment> segments2 = tm.connections.segmentsForCell({2});
        EXPECT_FALSE(segments2.empty());
        grewOnCell2 = true;
        segments.insert(segments.end(), segments2.begin(), segments2.end());
      }
      else
      {
        grewOnCell1 = true;
      }

      ASSERT_EQ(1, segments.size());
      vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
      EXPECT_EQ(4, synapses.size());

      set<Cell> columnChecklist(previousActiveColumns, previousActiveColumns+4);

      for (Synapse synapse : synapses)
      {
        SynapseData synapseData = tm.connections.dataForSynapse(synapse);
        EXPECT_NEAR(0.2, synapseData.permanence, EPSILON);

        UInt32 column = (UInt)tm.columnForCell(synapseData.presynapticCell);
        auto position = columnChecklist.find(column);
        EXPECT_NE(columnChecklist.end(), position);
        columnChecklist.erase(position);
      }
      EXPECT_TRUE(columnChecklist.empty());
    }

    EXPECT_TRUE(grewOnCell1);
    EXPECT_TRUE(grewOnCell2);
  }

  TEST(TemporalMemoryTest, testColumnForCell1D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    Cell cell;

    cell.idx = 0;
    ASSERT_TRUE(tm.columnForCell(cell) == 0);
    cell.idx = 4;
    ASSERT_TRUE(tm.columnForCell(cell) == 0);
    cell.idx = 5;
    ASSERT_TRUE(tm.columnForCell(cell) == 1);
    cell.idx = 10239;
    ASSERT_TRUE(tm.columnForCell(cell) == 2047);
  }

  TEST(TemporalMemoryTest, testColumnForCell2D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    Cell cell;

    cell.idx = 0;
    ASSERT_TRUE(tm.columnForCell(cell) == 0);
    cell.idx = 3;
    ASSERT_TRUE(tm.columnForCell(cell) == 0);
    cell.idx = 4;
    ASSERT_TRUE(tm.columnForCell(cell) == 1);
    cell.idx = 16383;
    ASSERT_TRUE(tm.columnForCell(cell) == 4095);
  }

  TEST(TemporalMemoryTest, testColumnForCellInvalidCell)
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

  TEST(TemporalMemoryTest, testCellsForColumn1D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    vector<Cell> expectedCells = { Cell(5), Cell(6), Cell(7), Cell(8), Cell(9) };
    vector<Cell> cellsForColumn = cellsForColumnCell(tm, 1);
    ASSERT_EQ(expectedCells, cellsForColumn);
  }

  TEST(TemporalMemoryTest, testCellsForColumn2D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);
    vector<Cell> expectedCells = { Cell(256), Cell(257), Cell(258), Cell(259) };
    vector<Cell> cellsForColumn = cellsForColumnCell(tm, 64);
    ASSERT_EQ(expectedCells, cellsForColumn);
  }

  TEST(TemporalMemoryTest, testNumberOfColumns)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    int numOfColumns = tm.numberOfColumns();
    ASSERT_EQ(numOfColumns, 64 * 64);
  }

  TEST(TemporalMemoryTest, testNumberOfCells)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    Int numberOfCells = tm.numberOfCells();
    ASSERT_EQ(numberOfCells, 64 * 64 * 32);
  }

  TEST(TemporalMemoryTest, testSaveLoad)
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

    check_tm_eq(tm1, tm2);

    int ret = ::remove(filename);
    ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
  }

  TEST(TemporalMemoryTest, testWrite)
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
    check_tm_eq(tm1, tm2);

    tm1.compute(sequence[0].size(), sequence[0].data());
    tm2.compute(sequence[0].size(), sequence[0].data());
    ASSERT_EQ(tm1.activeCells, tm2.activeCells);
    ASSERT_EQ(tm1.winnerCells, tm2.winnerCells);
    ASSERT_EQ(tm1.connections, tm2.connections);

    tm1.compute(sequence[3].size(), sequence[3].data());
    tm2.compute(sequence[3].size(), sequence[3].data());
    ASSERT_EQ(tm1.activeCells, tm2.activeCells);

    ASSERT_EQ(tm1.activeSegments.size(), tm2.activeSegments.size());
    for (size_t i = 0; i < tm1.activeSegments.size(); i++)
    {
      ASSERT_EQ(tm1.activeSegments[i].segment, tm2.activeSegments[i].segment);
      ASSERT_EQ(tm1.activeSegments[i].overlap, tm2.activeSegments[i].overlap);
    }

    ASSERT_EQ(tm1.matchingSegments.size(), tm2.matchingSegments.size());
    for (size_t i = 0; i < tm1.matchingSegments.size(); i++)
    {
      ASSERT_EQ(tm1.matchingSegments[i].segment, tm2.matchingSegments[i].segment);
      ASSERT_EQ(tm1.matchingSegments[i].overlap, tm2.matchingSegments[i].overlap);
    }

    ASSERT_EQ(tm1.winnerCells, tm2.winnerCells);
    ASSERT_EQ(tm1.connections, tm2.connections);

    check_tm_eq(tm1, tm2);
  }
} // end namespace nupic
