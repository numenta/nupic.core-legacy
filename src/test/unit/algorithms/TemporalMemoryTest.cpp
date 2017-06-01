/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
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

  /**
   * If you call compute with unsorted input, it should throw an exception.
   */
  TEST(TemporalMemoryTest, testCheckInputs_UnsortedColumns)
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

    const UInt activeColumns[4] = {1, 3, 2, 4};

    EXPECT_THROW(tm.compute(4, activeColumns), exception);
  }

  /**
   * If you call compute with a binary vector rather than a list of indices, it
   * should throw an exception.
   */
  TEST(TemporalMemoryTest, testCheckInputs_BinaryArray)
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

    // Use an input that will pass an `is_sorted` check.
    const UInt activeColumns[5] = {0, 0, 0, 1, 1};

    EXPECT_THROW(tm.compute(5, activeColumns), exception);
  }

  /**
   * When a predicted column is activated, only the predicted cells in the
   * columns should be activated.
   */
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

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment activeSegment =
      tm.createSegment(expectedActiveCells[0]);
    tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells());
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(expectedActiveCells, tm.getActiveCells());
  }

  /**
   * When an unpredicted column is activated, every cell in the column should
   * become active.
   */
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

    const UInt activeColumns[1] = {0};
    const vector<CellIdx> burstingCells = {0, 1, 2, 3};

    tm.compute(1, activeColumns, true);

    EXPECT_EQ(burstingCells, tm.getActiveCells());
  }

  /**
   * When the TemporalMemory receives zero active columns, it should still
   * compute the active cells, winner cells, and predictive cells. All should be
   * empty.
   */
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

    // Make some cells predictive.
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment segment = tm.createSegment(expectedActiveCells[0]);
    tm.connections.createSynapse(segment, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(segment, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(segment, previousActiveCells[2], 0.5);
    tm.connections.createSynapse(segment, previousActiveCells[3], 0.5);

    tm.compute(1, previousActiveColumns, true);
    ASSERT_FALSE(tm.getActiveCells().empty());
    ASSERT_FALSE(tm.getWinnerCells().empty());
    ASSERT_FALSE(tm.getPredictiveCells().empty());

    const UInt zeroColumns[0] = {};
    tm.compute(0, zeroColumns, true);

    EXPECT_TRUE(tm.getActiveCells().empty());
    EXPECT_TRUE(tm.getWinnerCells().empty());
    EXPECT_TRUE(tm.getPredictiveCells().empty());
  }

  /**
   * All predicted active cells are winner cells, even when learning is
   * disabled.
   */
  TEST(TemporalMemoryTest, PredictedActiveCellsAreAlwaysWinners)
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

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedWinnerCells = {4, 6};

    Segment activeSegment1 =
      tm.createSegment(expectedWinnerCells[0]);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment1, previousActiveCells[2], 0.5);

    Segment activeSegment2 =
      tm.createSegment(expectedWinnerCells[1]);
    tm.connections.createSynapse(activeSegment2, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment2, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment2, previousActiveCells[2], 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, false);
    tm.compute(numActiveColumns, activeColumns, false);

    EXPECT_EQ(expectedWinnerCells, tm.getWinnerCells());
  }

  /**
   * One cell in each bursting column is a winner cell, even when learning is
   * disabled.
   */
  TEST(TemporalMemoryTest, ChooseOneWinnerCellInBurstingColumn)
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

    const UInt activeColumns[1] = {0};
    const set<CellIdx> burstingCells = {0, 1, 2, 3};

    tm.compute(1, activeColumns, false);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    EXPECT_TRUE(burstingCells.find(winnerCells[0]) != burstingCells.end());
  }

  /**
   * Active segments on predicted active cells should be reinforced. Active
   * synapses should be reinforced, inactive synapses should be punished.
   */
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
      /*permanenceDecrement*/ 0.08,
      /*predictedSegmentDecrement*/ 0.02,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> activeCells = {5};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.createSegment(activeCell);
    Synapse activeSynapse1 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse =
      tm.connections.createSynapse(activeSegment, 81, 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.6, tm.connections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.connections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.connections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.42, tm.connections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * The best matching segment in a bursting column should be reinforced. Active
   * synapses should be strengthened, and inactive synapses should be weakened.
   */
  TEST(TemporalMemoryTest, ReinforceSelectedMatchingSegmentInBurstingColumn)
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
      /*permanenceDecrement*/ 0.08,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.createSegment(burstingCells[0]);
    Synapse activeSynapse1 =
      tm.connections.createSynapse(selectedMatchingSegment,
                                   previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.connections.createSynapse(selectedMatchingSegment,
                                   previousActiveCells[1], 0.3);
    Synapse activeSynapse3 =
      tm.connections.createSynapse(selectedMatchingSegment,
                                   previousActiveCells[2], 0.3);
    Synapse inactiveSynapse =
      tm.connections.createSynapse(selectedMatchingSegment,
                                   81, 0.3);

    // Add some competition.
    Segment otherMatchingSegment =
      tm.createSegment(burstingCells[1]);
    tm.connections.createSynapse(otherMatchingSegment,
                                 previousActiveCells[0], 0.3);
    tm.connections.createSynapse(otherMatchingSegment,
                                 previousActiveCells[1], 0.3);
    tm.connections.createSynapse(otherMatchingSegment,
                                 81, 0.3);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.4, tm.connections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.connections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.connections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.22, tm.connections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a column bursts, don't reward or punish matching-but-not-selected
   * segments.
   */
  TEST(TemporalMemoryTest, NoChangeToNonselectedMatchingSegmentsInBurstingColumn)
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
      /*permanenceDecrement*/ 0.08,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.createSegment(burstingCells[0]);
    tm.connections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[0], 0.3);
    tm.connections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[1], 0.3);
    tm.connections.createSynapse(selectedMatchingSegment,
                                 previousActiveCells[2], 0.3);
    tm.connections.createSynapse(selectedMatchingSegment,
                                 81, 0.3);

    Segment otherMatchingSegment =
      tm.createSegment(burstingCells[1]);
    Synapse activeSynapse1 =
      tm.connections.createSynapse(otherMatchingSegment,
                                   previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.connections.createSynapse(otherMatchingSegment,
                                   previousActiveCells[1], 0.3);
    Synapse inactiveSynapse =
      tm.connections.createSynapse(otherMatchingSegment,
                                   81, 0.3);

    tm.compute(1, previousActiveColumns, true);
    tm.compute(1, activeColumns, true);

    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a predicted column is activated, don't reward or punish
   * matching-but-not-active segments anywhere in the column.
   */
  TEST(TemporalMemoryTest, NoChangeToMatchingSegmentsInPredictedActiveColumn)
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

    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};
    const vector<CellIdx> otherBurstingCells = {5, 6, 7};

    Segment activeSegment =
      tm.createSegment(expectedActiveCells[0]);
    tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    Segment matchingSegmentOnSameCell =
      tm.createSegment(expectedActiveCells[0]);
    Synapse synapse1 =
      tm.connections.createSynapse(matchingSegmentOnSameCell,
                                   previousActiveCells[0], 0.3);
    Synapse synapse2 =
      tm.connections.createSynapse(matchingSegmentOnSameCell,
                                   previousActiveCells[1], 0.3);

    Segment matchingSegmentOnOtherCell =
      tm.createSegment(otherBurstingCells[0]);
    Synapse synapse3 =
      tm.connections.createSynapse(matchingSegmentOnOtherCell,
                                   previousActiveCells[0], 0.3);
    Synapse synapse4 =
      tm.connections.createSynapse(matchingSegmentOnOtherCell,
                                   previousActiveCells[1], 0.3);

    tm.compute(1, previousActiveColumns, true);
    ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells());
    tm.compute(1, activeColumns, true);

    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(synapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(synapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(synapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.connections.dataForSynapse(synapse4).permanence,
                EPSILON);
  }

  /**
   * When growing a new segment, if there are no previous winner cells, don't
   * even grow the segment. It will never match.
   */
  TEST(TemporalMemoryTest, NoNewSegmentIfNotEnoughWinnerCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 2,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt zeroColumns[0] = {};
    const UInt activeColumns[1] = {0};

    tm.compute(0, zeroColumns);
    tm.compute(1, activeColumns);

    EXPECT_EQ(0, tm.connections.numSegments());
  }

  /**
   * When growing a new segment, if the number of previous winner cells is above
   * maxNewSynapseCount, grow maxNewSynapseCount synapses.
   */
  TEST(TemporalMemoryTest, NewSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 2,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt previousActiveColumns[3] = {0, 1, 2};
    const UInt activeColumns[1] = {4};

    tm.compute(3, previousActiveColumns);

    vector<CellIdx> prevWinnerCells = tm.getWinnerCells();
    ASSERT_EQ(3, prevWinnerCells.size());

    tm.compute(1, activeColumns);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    vector<Segment> segments = tm.connections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
    ASSERT_EQ(2, synapses.size());
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.connections.dataForSynapse(synapse);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[0] ||
                  synapseData.presynapticCell == prevWinnerCells[1] ||
                  synapseData.presynapticCell == prevWinnerCells[2]);
    }

  }

  /**
   * When growing a new segment, if the number of previous winner cells is below
   * maxNewSynapseCount, grow synapses to all of the previous winner cells.
   */
  TEST(TemporalMemoryTest, NewSegmentAddSynapsesToAllWinnerCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    const UInt previousActiveColumns[3] = {0, 1, 2};
    const UInt activeColumns[1] = {4};

    tm.compute(3, previousActiveColumns);

    vector<CellIdx> prevWinnerCells = tm.getWinnerCells();
    ASSERT_EQ(3, prevWinnerCells.size());

    tm.compute(1, activeColumns);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    vector<Segment> segments = tm.connections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
    ASSERT_EQ(3, synapses.size());

    vector<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.connections.dataForSynapse(synapse);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      presynapticCells.push_back(synapseData.presynapticCell);
    }
    std::sort(presynapticCells.begin(), presynapticCells.end());
    EXPECT_EQ(prevWinnerCells, presynapticCells);
  }

  /**
   * When adding synapses to a matching segment, the final number of active
   * synapses on the segment should be maxNewSynapseCount, assuming there are
   * enough previous winner cells available to connect to.
   */
  TEST(TemporalMemoryTest, MatchingSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[4] = {0, 1, 2, 3};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.createSegment(4);
    tm.connections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(4, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.connections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(3, synapses.size());
    for (SynapseIdx i = 1; i < synapses.size(); i++)
    {
      SynapseData synapseData = tm.connections.dataForSynapse(synapses[i]);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[1] ||
                  synapseData.presynapticCell == prevWinnerCells[2] ||
                  synapseData.presynapticCell == prevWinnerCells[3]);
    }
  }

  /**
   * When adding synapses to a matching segment, if the number of previous
   * winner cells is lower than (maxNewSynapseCount - nActiveSynapsesOnSegment),
   * grow synapses to all the previous winner cells.
   */
  TEST(TemporalMemoryTest, MatchingSegmentAddSynapsesToAllWinnerCells)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[2] = {0, 1};
    const vector<CellIdx> prevWinnerCells = {0, 1};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.createSegment(4);
    tm.connections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(2, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.connections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(2, synapses.size());

    SynapseData synapseData = tm.connections.dataForSynapse(synapses[1]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_EQ(prevWinnerCells[1], synapseData.presynapticCell);
  }

  /**
   * When a segment becomes active, grow synapses to previous winner cells.
   *
   * The number of grown synapses is calculated from the "matching segment"
   * overlap, not the "active segment" overlap.
   */
  TEST(TemporalMemoryTest, ActiveSegmentGrowSynapsesAccordingToPotentialOverlap)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 2,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[5] = {0, 1, 2, 3, 4};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3, 4};
    const UInt activeColumns[1] = {5};

    Segment activeSegment = tm.createSegment(5);
    tm.connections.createSynapse(activeSegment, 0, 0.5);
    tm.connections.createSynapse(activeSegment, 1, 0.5);
    tm.connections.createSynapse(activeSegment, 2, 0.2);

    tm.compute(5, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.connections.synapsesForSegment(activeSegment);

    ASSERT_EQ(4, synapses.size());

    SynapseData synapseData = tm.connections.dataForSynapse(synapses[3]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[3] ||
                synapseData.presynapticCell == prevWinnerCells[4]);
  }

  /**
   * When a synapse is punished for contributing to a wrong prediction, if its
   * permanence falls to 0 it should be destroyed.
   */
  TEST(TemporalMemoryTest, DestroyWeakSynapseOnWrongPrediction)
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
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {2};
    const CellIdx expectedActiveCell = 5;

    Segment activeSegment = tm.createSegment(expectedActiveCell);
    tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);

    // Weak synapse.
    tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.015);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(3, tm.connections.numSynapses(activeSegment));
  }

  /**
   * When a synapse is punished for not contributing to a right prediction, if
   * its permanence falls to 0 it should be destroyed.
   */
  TEST(TemporalMemoryTest, DestroyWeakSynapseOnActiveReinforce)
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
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.createSegment(activeCell);
    tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);

    // Weak inactive synapse.
    tm.connections.createSynapse(activeSegment, 81, 0.09);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(3, tm.connections.numSynapses(activeSegment));
  }

  /**
   * When a segment adds synapses and it runs over maxSynapsesPerSegment, it
   * should make room by destroying synapses with the lowest permanence.
   */
  TEST(TemporalMemoryTest, RecycleWeakestSynapseToMakeRoomForNewSynapse)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 255,
      /*maxSynapsesPerSegment*/ 4
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[3] = {1, 2, 3};
    const vector<CellIdx> prevWinnerCells = {1, 2, 3};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.createSegment(4);

    // Create a weak synapse. Make sure it's not so weak that
    // permanenceDecrement destroys it.
    tm.connections.createSynapse(matchingSegment, 0, 0.11);

    // Create a synapse that will match.
    tm.connections.createSynapse(matchingSegment, 1, 0.20);

    // Create a synapse with a high permanence.
    tm.connections.createSynapse(matchingSegment, 31, 0.6);

    // Activate a synapse on the segment, making it "matching".
    tm.compute(3, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    // Now mark the segment as "correct" by activating its cell.
    tm.compute(1, activeColumns);

    // There should now be 3 synapses, and none of them should be to cell 0.
    const vector<Synapse>& synapses =
      tm.connections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(4, synapses.size());

    std::set<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      presynapticCells.insert(
        tm.connections.dataForSynapse(synapse).presynapticCell);
    }

    std::set<CellIdx> expected = {1, 2, 3, 31};
    EXPECT_EQ(expected, presynapticCells);
  }

  /**
   * When a cell adds a segment and it runs over maxSegmentsPerCell, it should
   * make room by destroying the least recently active segment.
   */
  TEST(TemporalMemoryTest, RecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    const UInt previousActiveColumns1[3] = {0, 1, 2};
    const UInt previousActiveColumns2[3] = {3, 4, 5};
    const UInt previousActiveColumns3[3] = {6, 7, 8};
    const UInt activeColumns[1] = {9};

    tm.compute(3, previousActiveColumns1);
    tm.compute(1, activeColumns);

    ASSERT_EQ(1, tm.connections.numSegments(9));
    Segment oldestSegment = tm.connections.segmentsForCell(9)[0];

    tm.reset();
    tm.compute(3, previousActiveColumns2);
    tm.compute(1, activeColumns);

    ASSERT_EQ(2, tm.connections.numSegments(9));

    set<CellIdx> oldPresynaptic;
    for (Synapse synapse : tm.connections.synapsesForSegment(oldestSegment))
    {
      oldPresynaptic.insert(
        tm.connections.dataForSynapse(synapse).presynapticCell);
    }

    tm.reset();
    tm.compute(3, previousActiveColumns3);
    tm.compute(1, activeColumns);

    ASSERT_EQ(2, tm.connections.numSegments(9));

    // Verify none of the segments are connected to the cells the old segment
    // was connected to.

    for (Segment segment : tm.connections.segmentsForCell(9))
    {
      set<CellIdx> newPresynaptic;
      for (Synapse synapse : tm.connections.synapsesForSegment(segment))
      {
        newPresynaptic.insert(
          tm.connections.dataForSynapse(synapse).presynapticCell);
      }

      vector<CellIdx> intersection;
      std::set_intersection(oldPresynaptic.begin(), oldPresynaptic.end(),
                            newPresynaptic.begin(), newPresynaptic.end(),
                            std::back_inserter(intersection));

      vector<CellIdx> expected = {};
      EXPECT_EQ(expected, intersection);
    }
  }

  /**
   * When a segment's number of synapses falls to 0, the segment should be
   * destroyed.
   */
  TEST(TemporalMemoryTest, DestroySegmentsWithTooFewSynapsesToBeMatching)
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
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {2};
    const CellIdx expectedActiveCell = 5;

    Segment matchingSegment = tm.createSegment(expectedActiveCell);
    tm.connections.createSynapse(matchingSegment, previousActiveCells[0], 0.015);
    tm.connections.createSynapse(matchingSegment, previousActiveCells[1], 0.015);
    tm.connections.createSynapse(matchingSegment, previousActiveCells[2], 0.015);
    tm.connections.createSynapse(matchingSegment, previousActiveCells[3], 0.015);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(0, tm.connections.numSegments(expectedActiveCell));
  }

  /**
   * When a column with a matching segment isn't activated, punish the matching
   * segment.
   *
   * To exercise the implementation:
   *
   *  - Use cells before, between, and after the active columns.
   *  - Use segments that are matching-but-not-active and matching-and-active.
   */
  TEST(TemporalMemoryTest, PunishMatchingSegmentsInInactiveColumns)
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
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const CellIdx previousInactiveCell = 81;

    Segment activeSegment = tm.createSegment(42);
    Synapse activeSynapse1 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse1 =
      tm.connections.createSynapse(activeSegment, previousInactiveCell, 0.5);

    Segment matchingSegment = tm.createSegment(43);
    Synapse activeSynapse4 =
      tm.connections.createSynapse(matchingSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse5 =
      tm.connections.createSynapse(matchingSegment, previousActiveCells[1], 0.5);
    Synapse inactiveSynapse2 =
      tm.connections.createSynapse(matchingSegment, previousInactiveCell, 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.48, tm.connections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.connections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.connections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.connections.dataForSynapse(activeSynapse4).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.connections.dataForSynapse(activeSynapse5).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.connections.dataForSynapse(inactiveSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.connections.dataForSynapse(inactiveSynapse2).permanence,
                EPSILON);
  }

  /**
   * In a bursting column with no matching segments, a segment should be added
   * to the cell with the fewest segments. When there's a tie, choose randomly.
   */
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
      const vector<CellIdx> previousActiveCells =
        {4, 5, 6, 7}; // (there are more)
      vector<CellIdx> nonmatchingCells = {0, 3};
      vector<CellIdx> activeCells = {0, 1, 2, 3};

      Segment segment1 = tm.createSegment(nonmatchingCells[0]);
      tm.connections.createSynapse(segment1, previousActiveCells[0], 0.5);
      Segment segment2 = tm.createSegment(nonmatchingCells[1]);
      tm.connections.createSynapse(segment2, previousActiveCells[1], 0.5);

      tm.compute(4, previousActiveColumns, true);
      tm.compute(1, activeColumns, true);

      ASSERT_EQ(activeCells, tm.getActiveCells());

      EXPECT_EQ(3, tm.connections.numSegments());
      EXPECT_EQ(1, tm.connections.segmentsForCell(0).size());
      EXPECT_EQ(1, tm.connections.segmentsForCell(3).size());
      EXPECT_EQ(1, tm.connections.numSynapses(segment1));
      EXPECT_EQ(1, tm.connections.numSynapses(segment2));

      vector<Segment> segments = tm.connections.segmentsForCell(1);
      if (segments.empty())
      {
        vector<Segment> segments2 = tm.connections.segmentsForCell(2);
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

      set<CellIdx> columnChecklist(previousActiveColumns, previousActiveColumns+4);

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

  /**
   * When the best matching segment has more than maxNewSynapseCount matching
   * synapses, don't grow new synapses. This test is specifically aimed at
   * unexpected behavior with negative numbers and unsigned integers.
   */
  TEST(TemporalMemoryTest, MaxNewSynapseCountOverflow)
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

    Segment segment = tm.createSegment(8);
    tm.connections.createSynapse(segment, 0, 0.2);
    tm.connections.createSynapse(segment, 1, 0.2);
    tm.connections.createSynapse(segment, 2, 0.2);
    tm.connections.createSynapse(segment, 3, 0.2);
    tm.connections.createSynapse(segment, 4, 0.2);
    Synapse sampleSynapse = tm.connections.createSynapse(segment, 5, 0.2);
    tm.connections.createSynapse(segment, 6, 0.2);
    tm.connections.createSynapse(segment, 7, 0.2);

    const UInt previousActiveColumns[4] = {0, 1, 3, 4};
    tm.compute(4, previousActiveColumns);

    ASSERT_EQ(1, tm.getMatchingSegments().size());

    const UInt activeColumns[1] = {2};
    tm.compute(1, activeColumns);

    // Make sure the segment has learned.
    ASSERT_NEAR(0.3, tm.connections.dataForSynapse(sampleSynapse).permanence,
                EPSILON);

    EXPECT_EQ(8, tm.connections.numSynapses(segment));
  }

  /**
   * With learning disabled, generate some predicted active columns, predicted
   * inactive columns, and nonpredicted active columns. The connections should
   * not change.
   */
  TEST(TemporalMemoryTest, ConnectionsNeverChangeWhenLearningDisabled)
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

    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[2] = {
      1, // predicted
      2  // bursting
    };
    const CellIdx previousInactiveCell = 81;
    const vector<CellIdx> expectedActiveCells = {4};

    Segment correctActiveSegment =
      tm.createSegment(expectedActiveCells[0]);
    tm.connections.createSynapse(correctActiveSegment,
                                 previousActiveCells[0], 0.5);
    tm.connections.createSynapse(correctActiveSegment,
                                 previousActiveCells[1], 0.5);
    tm.connections.createSynapse(correctActiveSegment,
                                 previousActiveCells[2], 0.5);

    Segment wrongMatchingSegment = tm.createSegment(43);
    tm.connections.createSynapse(wrongMatchingSegment,
                                 previousActiveCells[0], 0.5);
    tm.connections.createSynapse(wrongMatchingSegment,
                                 previousActiveCells[1], 0.5);
    tm.connections.createSynapse(wrongMatchingSegment,
                                 previousInactiveCell, 0.5);

    Connections before = tm.connections;

    tm.compute(1, previousActiveColumns, false);
    tm.compute(2, activeColumns, false);

    EXPECT_EQ(before, tm.connections);
  }

   /**
   * Destroy some segments then verify that the maxSegmentsPerCell is still
   * correctly applied.
   */
  TEST(TemporalMemoryTest, DestroySegmentsThenReachLimit)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    {
      Segment segment1 = tm.createSegment(11);
      Segment segment2 = tm.createSegment(11);
      ASSERT_EQ(2, tm.connections.numSegments());
      tm.connections.destroySegment(segment1);
      tm.connections.destroySegment(segment2);
      ASSERT_EQ(0, tm.connections.numSegments());
    }

    {
      tm.createSegment(11);
      EXPECT_EQ(1, tm.connections.numSegments());
      tm.createSegment(11);
      EXPECT_EQ(2, tm.connections.numSegments());
      tm.createSegment(11);
      EXPECT_EQ(2, tm.connections.numSegments());
      EXPECT_EQ(2, tm.connections.numSegments(11));
    }
  }

   /**
   * Creates many segments on a cell, until hits segment limit. Then creates
   * another segment, and checks that it destroyed the least recently used
   * segment and created a new one in its place.
   */
  TEST(TemporalMemoryTest, CreateSegmentDestroyOld)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    Segment segment1 = tm.createSegment(12);

    tm.connections.createSynapse(segment1, 1, 0.5);
    tm.connections.createSynapse(segment1, 2, 0.5);
    tm.connections.createSynapse(segment1, 3, 0.5);

    // Let some time pass.
    tm.compute(0, nullptr);
    tm.compute(0, nullptr);
    tm.compute(0, nullptr);

    // Create a segment with 1 synapse.
    Segment segment2 = tm.createSegment(12);
    tm.connections.createSynapse(segment2, 1, 0.5);

    tm.compute(0, nullptr);

    // Give the first segment some activity.
    const UInt activeColumns[3] = {1, 2, 3};
    tm.compute(3, activeColumns);

    // Create a new segment with no synapses.
    tm.createSegment(12);

    vector<Segment> segments = tm.connections.segmentsForCell(12);
    ASSERT_EQ(2, segments.size());

    // Verify first segment is still there with the same synapses.
    vector<Synapse> synapses1 = tm.connections.synapsesForSegment(segments[0]);
    ASSERT_EQ(3, synapses1.size());
    ASSERT_EQ(1, tm.connections.dataForSynapse(synapses1[0]).presynapticCell);
    ASSERT_EQ(2, tm.connections.dataForSynapse(synapses1[1]).presynapticCell);
    ASSERT_EQ(3, tm.connections.dataForSynapse(synapses1[2]).presynapticCell);

    // Verify second segment has been replaced.
    ASSERT_EQ(0, tm.connections.numSynapses(segments[1]));
  }

   /**
    * Hit the maxSegmentsPerCell threshold multiple times. Make sure it works
    * more than once.
    */
  TEST(ConnectionsTest, ReachSegmentLimitMultipleTimes)
  {
    TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*predictedSegmentDecrement*/ 0.0,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    tm.createSegment(10);
    ASSERT_EQ(1, tm.connections.numSegments());
    tm.createSegment(10);
    ASSERT_EQ(2, tm.connections.numSegments());
    tm.createSegment(10);
    ASSERT_EQ(2, tm.connections.numSegments());
    tm.createSegment(10);
    EXPECT_EQ(2, tm.connections.numSegments());
  }

  TEST(TemporalMemoryTest, testColumnForCell1D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    ASSERT_EQ(0, tm.columnForCell(0));
    ASSERT_EQ(0, tm.columnForCell(4));
    ASSERT_EQ(1, tm.columnForCell(5));
    ASSERT_EQ(2047, tm.columnForCell(10239));
  }

  TEST(TemporalMemoryTest, testColumnForCell2D)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    ASSERT_EQ(0, tm.columnForCell(0));
    ASSERT_EQ(0, tm.columnForCell(3));
    ASSERT_EQ(1, tm.columnForCell(4));
    ASSERT_EQ(4095, tm.columnForCell(16383));
  }

  TEST(TemporalMemoryTest, testColumnForCellInvalidCell)
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    EXPECT_NO_THROW(tm.columnForCell(16383));
    EXPECT_THROW(tm.columnForCell(16384), std::exception);
    EXPECT_THROW(tm.columnForCell(-1), std::exception);
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

  void serializationTestPrepare(TemporalMemory& tm)
  {
    // Create an active segment and a two matching segments.
    // Destroy a few to exercise the code.
    Segment destroyMe1 = tm.createSegment(4);
    tm.connections.destroySegment(destroyMe1);

    Segment activeSegment = tm.createSegment(4);
    tm.connections.createSynapse(activeSegment, 0, 0.5);
    tm.connections.createSynapse(activeSegment, 1, 0.5);
    Synapse destroyMe2 = tm.connections.createSynapse(activeSegment, 42, 0.5);
    tm.connections.destroySynapse(destroyMe2);
    tm.connections.createSynapse(activeSegment, 2, 0.5);
    tm.connections.createSynapse(activeSegment, 3, 0.5);

    Segment matchingSegment1 = tm.createSegment(8);
    tm.connections.createSynapse(matchingSegment1, 0, 0.4);
    tm.connections.createSynapse(matchingSegment1, 1, 0.4);
    tm.connections.createSynapse(matchingSegment1, 2, 0.4);

    Segment matchingSegment2 = tm.createSegment(9);
    tm.connections.createSynapse(matchingSegment2, 0, 0.4);
    tm.connections.createSynapse(matchingSegment2, 1, 0.4);
    tm.connections.createSynapse(matchingSegment2, 2, 0.4);
    tm.connections.createSynapse(matchingSegment2, 3, 0.4);

    UInt activeColumns[] = {0};
    tm.compute(1, activeColumns);

    ASSERT_EQ(1, tm.getActiveSegments().size());
    ASSERT_EQ(3, tm.getMatchingSegments().size());
  }

  void serializationTestVerify(TemporalMemory& tm)
  {
    // Activate 3 columns. One has an active segment, one has two
    // matching segments, and one has none. One column should be
    // predicted, the others should burst, there should be four
    // segments total, and they should have the correct permanences
    // and synapse counts.

    const vector<UInt> prevWinnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, prevWinnerCells.size());

    UInt activeColumns[] = {1, 2, 3};
    tm.compute(3, activeColumns);

    // Verify the correct cells were activated.
    EXPECT_EQ((vector<UInt>{4, 8, 9, 10, 11, 12, 13, 14, 15}),
              tm.getActiveCells());
    const vector<UInt> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(3, winnerCells.size());
    EXPECT_EQ(4, winnerCells[0]);
    EXPECT_EQ(9, winnerCells[1]);

    EXPECT_EQ(4, tm.connections.numSegments());

    // Verify the active segment learned.
    ASSERT_EQ(1, tm.connections.numSegments(4));
    Segment activeSegment = tm.connections.segmentsForCell(4)[0];
    const vector<Synapse> syns1 =
      tm.connections.synapsesForSegment(activeSegment);
    ASSERT_EQ(4, syns1.size());
    EXPECT_EQ(0,
              tm.connections.dataForSynapse(syns1[0]).presynapticCell);
    EXPECT_NEAR(0.6,
                tm.connections.dataForSynapse(syns1[0]).permanence,
                EPSILON);
    EXPECT_EQ(1,
              tm.connections.dataForSynapse(syns1[1]).presynapticCell);
    EXPECT_NEAR(0.6,
                tm.connections.dataForSynapse(syns1[1]).permanence,
                EPSILON);
    EXPECT_EQ(2,
              tm.connections.dataForSynapse(syns1[2]).presynapticCell);
    EXPECT_NEAR(0.6,
                tm.connections.dataForSynapse(syns1[2]).permanence,
                EPSILON);
    EXPECT_EQ(3,
              tm.connections.dataForSynapse(syns1[3]).presynapticCell);
    EXPECT_NEAR(0.6,
                tm.connections.dataForSynapse(syns1[3]).permanence,
                EPSILON);

    // Verify the non-best matching segment is unchanged.
    ASSERT_EQ(1, tm.connections.numSegments(8));
    Segment matchingSegment1 = tm.connections.segmentsForCell(8)[0];
    const vector<Synapse> syns2 =
      tm.connections.synapsesForSegment(matchingSegment1);
    ASSERT_EQ(3, syns2.size());
    EXPECT_EQ(0,
              tm.connections.dataForSynapse(syns2[0]).presynapticCell);
    EXPECT_NEAR(0.4,
                tm.connections.dataForSynapse(syns2[0]).permanence,
                EPSILON);
    EXPECT_EQ(1,
              tm.connections.dataForSynapse(syns2[1]).presynapticCell);
    EXPECT_NEAR(0.4,
                tm.connections.dataForSynapse(syns2[1]).permanence,
                EPSILON);
    EXPECT_EQ(2,
              tm.connections.dataForSynapse(syns2[2]).presynapticCell);
    EXPECT_NEAR(0.4,
                tm.connections.dataForSynapse(syns2[2]).permanence,
                EPSILON);

    // Verify the best matching segment learned.
    ASSERT_EQ(1, tm.connections.numSegments(9));
    Segment matchingSegment2 = tm.connections.segmentsForCell(9)[0];
    const vector<Synapse> syns3 =
      tm.connections.synapsesForSegment(matchingSegment2);
    ASSERT_EQ(4, syns3.size());
    EXPECT_EQ(0,
              tm.connections.dataForSynapse(syns3[0]).presynapticCell);
    EXPECT_NEAR(0.5,
                tm.connections.dataForSynapse(syns3[0]).permanence,
                EPSILON);
    EXPECT_EQ(1,
              tm.connections.dataForSynapse(syns3[1]).presynapticCell);
    EXPECT_NEAR(0.5,
                tm.connections.dataForSynapse(syns3[1]).permanence,
                EPSILON);
    EXPECT_EQ(2,
              tm.connections.dataForSynapse(syns3[2]).presynapticCell);
    EXPECT_NEAR(0.5,
                tm.connections.dataForSynapse(syns3[2]).permanence,
                EPSILON);
    EXPECT_EQ(3,
              tm.connections.dataForSynapse(syns3[3]).presynapticCell);
    EXPECT_NEAR(0.5,
                tm.connections.dataForSynapse(syns3[3]).permanence,
                EPSILON);

    // Verify the winner cell in the last column grew a segment.
    const UInt winnerCell = winnerCells[2];
    EXPECT_GE(winnerCell, 12);
    EXPECT_LT(winnerCell, 16);
    ASSERT_EQ(1, tm.connections.numSegments(winnerCell));
    Segment newSegment = tm.connections.segmentsForCell(winnerCell)[0];
    const vector<Synapse> syns4 =
      tm.connections.synapsesForSegment(newSegment);
    ASSERT_EQ(1, syns4.size());
    EXPECT_EQ(prevWinnerCells[0],
              tm.connections.dataForSynapse(syns4[0]).presynapticCell);
    EXPECT_NEAR(0.21,
                tm.connections.dataForSynapse(syns4[0]).permanence,
                EPSILON);
  }

  TEST(TemporalMemoryTest, testSaveLoad)
  {
    TemporalMemory tm1(
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

    serializationTestPrepare(tm1);

    stringstream ss;
    tm1.save(ss);

    TemporalMemory tm2;
    tm2.load(ss);

    ASSERT_TRUE(tm1 == tm2);

    serializationTestVerify(tm2);
  }

  TEST(TemporalMemoryTest, testWrite)
  {
    TemporalMemory tm1(
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

    serializationTestPrepare(tm1);

    // Write and read back the proto
    stringstream ss;
    tm1.write(ss);

    TemporalMemory tm2;
    tm2.read(ss);

    ASSERT_TRUE(tm1 == tm2);

    serializationTestVerify(tm2);
  }

  // Uncomment these tests individually to save/load from a file.
  // This is useful for ad-hoc testing of backwards-compatibility.

  // TEST(TemporalMemoryTest, saveTestFile)
  // {
  //   TemporalMemory tm(
  //     /*columnDimensions*/ {32},
  //     /*cellsPerColumn*/ 4,
  //     /*activationThreshold*/ 3,
  //     /*initialPermanence*/ 0.21,
  //     /*connectedPermanence*/ 0.50,
  //     /*minThreshold*/ 2,
  //     /*maxNewSynapseCount*/ 3,
  //     /*permanenceIncrement*/ 0.10,
  //     /*permanenceDecrement*/ 0.10,
  //     /*predictedSegmentDecrement*/ 0.0,
  //     /*seed*/ 42
  //     );
  //
  //   serializationTestPrepare(tm);
  //
  //   const char* filename = "TemporalMemorySerializationSave.tmp";
  //   ofstream outfile;
  //   outfile.open(filename, ios::binary);
  //   tm.save(outfile);
  //   outfile.close();
  // }

  // TEST(TemporalMemoryTest, loadTestFile)
  // {
  //   TemporalMemory tm;
  //   const char* filename = "TemporalMemorySerializationSave.tmp";
  //   ifstream infile(filename, ios::binary);
  //   tm.load(infile);
  //   infile.close();
  //
  //   serializationTestVerify(tm);
  // }

  // TEST(TemporalMemoryTest, writeTestFile)
  // {
  //   TemporalMemory tm(
  //     /*columnDimensions*/ {32},
  //     /*cellsPerColumn*/ 4,
  //     /*activationThreshold*/ 3,
  //     /*initialPermanence*/ 0.21,
  //     /*connectedPermanence*/ 0.50,
  //     /*minThreshold*/ 2,
  //     /*maxNewSynapseCount*/ 3,
  //     /*permanenceIncrement*/ 0.10,
  //     /*permanenceDecrement*/ 0.10,
  //     /*predictedSegmentDecrement*/ 0.0,
  //     /*seed*/ 42
  //     );

  //   serializationTestPrepare(tm);

  //   const char* filename = "TemporalMemorySerializationWrite.tmp";
  //   ofstream outfile;
  //   outfile.open(filename, ios::binary);
  //   tm.write(outfile);
  //   outfile.close();
  // }

  // TEST(TemporalMemoryTest, readTestFile)
  // {
  //   TemporalMemory tm;
  //   const char* filename = "TemporalMemorySerializationWrite.tmp";
  //   ifstream infile(filename, ios::binary);
  //   tm.read(infile);
  //   infile.close();

  //   serializationTestVerify(tm);
  // }
} // end namespace nupic
