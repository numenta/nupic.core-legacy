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
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for ExtendedTemporalMemory
 */

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

#include <nupic/experimental/ExtendedTemporalMemory.hpp>
#include "gtest/gtest.h"

using namespace nupic::experimental::extended_temporal_memory;
using namespace std;

#define EPSILON 0.0000001

namespace {
  void check_tm_eq(const ExtendedTemporalMemory& tm1,
                   const ExtendedTemporalMemory& tm2)
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

  TEST(ExtendedTemporalMemoryTest, testInitInvalidParams)
  {
    // Invalid columnDimensions
    vector<UInt> columnDim = {};
    ExtendedTemporalMemory tm1;
    EXPECT_THROW(tm1.initialize(columnDim, 32), exception);

    // Invalid cellsPerColumn
    columnDim.push_back(2048);
    EXPECT_THROW(tm1.initialize(columnDim, 0), exception);
  }

  /**
   * When a predicted column is activated, only the predicted cells in the
   * columns should be activated.
   */
  TEST(ExtendedTemporalMemoryTest, ActivateCorrectlyPredictiveCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment activeSegment =
      tm.basalConnections.createSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells());
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(expectedActiveCells, tm.getActiveCells());
  }

  /**
   * When an unpredicted column is activated, every cell in the column should
   * become active.
   */
  TEST(ExtendedTemporalMemoryTest, BurstUnpredictedColumns)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt activeColumns[1] = {0};
    const vector<CellIdx> burstingCells = {0, 1, 2, 3};

    tm.compute(1, activeColumns, true);

    EXPECT_EQ(burstingCells, tm.getActiveCells());
  }

  /**
   * When the ExtendedTemporalMemory receives zero active columns, it should still
   * compute the active cells, winner cells, and predictive cells. All should be
   * empty.
   */
  TEST(ExtendedTemporalMemoryTest, ZeroActiveColumns)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Make some cells predictive.
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment segment = tm.basalConnections.createSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(segment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[3], 0.5);

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
  TEST(ExtendedTemporalMemoryTest, PredictedActiveCellsAreAlwaysWinners)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedWinnerCells = {4, 6};

    Segment activeSegment1 =
      tm.basalConnections.createSegment(expectedWinnerCells[0]);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[2], 0.5);

    Segment activeSegment2 =
      tm.basalConnections.createSegment(expectedWinnerCells[1]);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[2], 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, false);
    tm.compute(numActiveColumns, activeColumns, false);

    EXPECT_EQ(expectedWinnerCells, tm.getWinnerCells());
  }

  /**
   * One cell in each bursting column is a winner cell, even when learning is
   * disabled.
   */
  TEST(ExtendedTemporalMemoryTest, ChooseOneWinnerCellInBurstingColumn)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
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
  TEST(ExtendedTemporalMemoryTest, ReinforceCorrectlyActiveSegments)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> activeCells = {5};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.basalConnections.createSegment(activeCell);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(activeSegment, 81, 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.42, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * Active segments on predicted active cells should not grow new synapses.
   */
  TEST(ExtendedTemporalMemoryTest, NoGrowthOnCorrectlyActiveSegments)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> activeCells = {5};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.basalConnections.createSegment(activeCell);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_EQ(3, tm.basalConnections.numSynapses(activeSegment));
  }

  /**
   * The best matching segment in a bursting column should be reinforced. Active
   * synapses should be strengthened, and inactive synapses should be weakened.
   */
  TEST(ExtendedTemporalMemoryTest, ReinforceSelectedMatchingSegmentInBurstingColumn)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.basalConnections.createSegment(burstingCells[0]);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[1], 0.3);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[2], 0.3);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        81, 0.3);

    // Add some competition.
    Segment otherMatchingSegment =
      tm.basalConnections.createSegment(burstingCells[1]);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      previousActiveCells[0], 0.3);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      previousActiveCells[1], 0.3);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      81, 0.3);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.22, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a column bursts, don't reward or punish matching-but-not-selected
   * segments.
   */
  TEST(ExtendedTemporalMemoryTest, NoChangeToNonselectedMatchingSegmentsInBurstingColumn)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.basalConnections.createSegment(burstingCells[0]);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[0], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[1], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[2], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      81, 0.3);

    Segment otherMatchingSegment =
      tm.basalConnections.createSegment(burstingCells[1]);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        previousActiveCells[1], 0.3);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        81, 0.3);

    tm.compute(1, previousActiveColumns, true);
    tm.compute(1, activeColumns, true);

    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a predicted column is activated, don't reward or punish
   * matching-but-not-active segments anywhere in the column.
   */
  TEST(ExtendedTemporalMemoryTest, NoChangeToMatchingSegmentsInPredictedActiveColumn)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt previousActiveColumns[1] = {0};
    const UInt activeColumns[1] = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};
    const vector<CellIdx> otherBurstingCells = {5, 6, 7};

    Segment activeSegment =
      tm.basalConnections.createSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    Segment matchingSegmentOnSameCell =
      tm.basalConnections.createSegment(expectedActiveCells[0]);
    Synapse synapse1 =
      tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                        previousActiveCells[0], 0.3);
    Synapse synapse2 =
      tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                        previousActiveCells[1], 0.3);

    Segment matchingSegmentOnOtherCell =
      tm.basalConnections.createSegment(otherBurstingCells[0]);
    Synapse synapse3 =
      tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                        previousActiveCells[0], 0.3);
    Synapse synapse4 =
      tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                        previousActiveCells[1], 0.3);

    tm.compute(1, previousActiveColumns, true);
    ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells());
    tm.compute(1, activeColumns, true);

    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse4).permanence,
                EPSILON);
  }

  /**
   * When growing a new segment, if there are no previous winner cells, don't
   * even grow the segment. It will never match.
   */
  TEST(ExtendedTemporalMemoryTest, NoNewSegmentIfNotEnoughWinnerCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt zeroColumns[0] = {};
    const UInt activeColumns[1] = {0};

    tm.compute(0, zeroColumns);
    tm.compute(1, activeColumns);

    EXPECT_EQ(0, tm.basalConnections.numSegments());
  }

  /**
   * When growing a new segment, if the number of previous winner cells is above
   * maxNewSynapseCount, grow maxNewSynapseCount synapses.
   */
  TEST(ExtendedTemporalMemoryTest, NewSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
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
    vector<Segment> segments = tm.basalConnections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
    ASSERT_EQ(2, synapses.size());
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
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
  TEST(ExtendedTemporalMemoryTest, NewSegmentAddSynapsesToAllWinnerCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
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
    vector<Segment> segments = tm.basalConnections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
    ASSERT_EQ(3, synapses.size());

    vector<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
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
  TEST(ExtendedTemporalMemoryTest, MatchingSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[4] = {0, 1, 2, 3};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.basalConnections.createSegment(4);
    tm.basalConnections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(4, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(3, synapses.size());
    for (SynapseIdx i = 1; i < synapses.size(); i++)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[i]);
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
  TEST(ExtendedTemporalMemoryTest, MatchingSegmentAddSynapsesToAllWinnerCells)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[2] = {0, 1};
    const vector<CellIdx> prevWinnerCells = {0, 1};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.basalConnections.createSegment(4);
    tm.basalConnections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(2, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(2, synapses.size());

    SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[1]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_EQ(prevWinnerCells[1], synapseData.presynapticCell);
  }

  /**
   * When a segment becomes active, grow synapses to previous winner cells.
   *
   * The number of grown synapses is calculated from the "matching segment"
   * overlap, not the "active segment" overlap.
   */
  TEST(ExtendedTemporalMemoryTest, ActiveSegmentGrowSynapsesAccordingToPotentialOverlap)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[5] = {0, 1, 2, 3, 4};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3, 4};
    const UInt activeColumns[1] = {5};

    Segment activeSegment = tm.basalConnections.createSegment(5);
    tm.basalConnections.createSynapse(activeSegment, 0, 0.5);
    tm.basalConnections.createSynapse(activeSegment, 1, 0.5);
    tm.basalConnections.createSynapse(activeSegment, 2, 0.2);

    tm.compute(5, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(activeSegment);

    ASSERT_EQ(4, synapses.size());

    SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[3]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[3] ||
                synapseData.presynapticCell == prevWinnerCells[4]);
  }

  /**
   * When a synapse is punished for contributing to a wrong prediction, if its
   * permanence falls to 0 it should be destroyed.
   */
  TEST(ExtendedTemporalMemoryTest, DestroyWeakSynapseOnWrongPrediction)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {2};
    const CellIdx expectedActiveCell = 5;

    Segment activeSegment = tm.basalConnections.createSegment(expectedActiveCell);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse weakActiveSynapse =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.015);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_TRUE(tm.basalConnections.dataForSynapse(weakActiveSynapse).destroyed);
  }

  /**
   * When a synapse is punished for not contributing to a right prediction, if
   * its permanence falls to 0 it should be destroyed.
   */
  TEST(ExtendedTemporalMemoryTest, DestroyWeakSynapseOnActiveReinforce)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.basalConnections.createSegment(activeCell);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse weakInactiveSynapse =
      tm.basalConnections.createSynapse(activeSegment, 81, 0.09);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_TRUE(tm.basalConnections.dataForSynapse(weakInactiveSynapse).destroyed);
  }

  /**
   * When a segment adds synapses and it runs over maxSynapsesPerSegment, it
   * should make room by destroying synapses with the lowest permanence.
   */
  TEST(ExtendedTemporalMemoryTest, RecycleWeakestSynapseToMakeRoomForNewSynapse)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 255,
      /*maxSynapsesPerSegment*/ 3
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const UInt previousActiveColumns[3] = {0, 1, 2};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2};
    const UInt activeColumns[1] = {4};

    Segment matchingSegment = tm.basalConnections.createSegment(4);
    tm.basalConnections.createSynapse(matchingSegment, 81, 0.6);

    // Still the weakest after adding permanenceIncrement.
    Synapse weakestSynapse =
      tm.basalConnections.createSynapse(matchingSegment, 0, 0.11);

    tm.compute(3, previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(1, activeColumns);

    // Note that it destroys the weak active synapse, not the strong inactive
    // synapse.
    SynapseData synapseData = tm.basalConnections.dataForSynapse(weakestSynapse);
    EXPECT_NE(0, synapseData.presynapticCell);
    EXPECT_FALSE(synapseData.destroyed);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
  }

  /**
   * When a cell adds a segment and it runs over maxSegmentsPerCell, it should
   * make room by destroying the least recently active segment.
   */
  TEST(ExtendedTemporalMemoryTest, RecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    const UInt previousActiveColumns1[3] = {0, 1, 2};
    const UInt previousActiveColumns2[3] = {3, 4, 5};
    const UInt previousActiveColumns3[3] = {6, 7, 8};
    const UInt activeColumns[1] = {9};

    tm.compute(3, previousActiveColumns1);
    tm.compute(1, activeColumns);

    ASSERT_EQ(1, tm.basalConnections.numSegments(9));
    Segment oldestSegment = tm.basalConnections.segmentsForCell(9)[0];

    tm.reset();
    tm.compute(3, previousActiveColumns2);
    tm.compute(1, activeColumns);

    ASSERT_EQ(2, tm.basalConnections.numSegments(9));

    tm.reset();
    tm.compute(3, previousActiveColumns3);
    tm.compute(1, activeColumns);

    ASSERT_EQ(2, tm.basalConnections.numSegments(9));

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(oldestSegment);
    ASSERT_EQ(3, synapses.size());
    set<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
      presynapticCells.insert(synapseData.presynapticCell);
    }

    const set<CellIdx> expected = {6, 7, 8};
    EXPECT_EQ(expected, presynapticCells);
  }

  /**
   * When a segment's number of synapses falls to 0, the segment should be
   * destroyed.
   */
  TEST(ExtendedTemporalMemoryTest, DestroySegmentsWithTooFewSynapsesToBeMatching)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {2};
    const CellIdx expectedActiveCell = 5;

    Segment matchingSegment = tm.basalConnections.createSegment(expectedActiveCell);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[0], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[1], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[2], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[3], 0.015);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_TRUE(tm.basalConnections.dataForSegment(matchingSegment).destroyed);
    EXPECT_EQ(0, tm.basalConnections.numSegments(expectedActiveCell));
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
  TEST(ExtendedTemporalMemoryTest, PunishMatchingSegmentsInInactiveColumns)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const UInt activeColumns[1] = {1};
    const CellIdx previousInactiveCell = 81;

    Segment activeSegment = tm.basalConnections.createSegment(42);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousInactiveCell, 0.5);

    Segment matchingSegment = tm.basalConnections.createSegment(43);
    Synapse activeSynapse4 =
      tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse5 =
      tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[1], 0.5);
    Synapse inactiveSynapse2 =
      tm.basalConnections.createSynapse(matchingSegment, previousInactiveCell, 0.5);

    tm.compute(numActiveColumns, previousActiveColumns, true);
    tm.compute(numActiveColumns, activeColumns, true);

    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse4).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse5).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.basalConnections.dataForSynapse(inactiveSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.basalConnections.dataForSynapse(inactiveSynapse2).permanence,
                EPSILON);
  }

  /**
   * In a bursting column with no matching segments, a segment should be added
   * to the cell with the fewest segments. When there's a tie, choose randomly.
   */
  TEST(ExtendedTemporalMemoryTest, AddSegmentToCellWithFewestSegments)
  {
    bool grewOnCell1 = false;
    bool grewOnCell2 = false;
    for (UInt seed = 0; seed < 100; seed++)
    {
      ExtendedTemporalMemory tm(
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
        /*formInternalBasalConnections*/ true,
        /*learnOnOneCell*/ false,
        /*seed*/ seed
        );

      // enough for 4 winner cells
      const UInt previousActiveColumns[4] ={1, 2, 3, 4};
      const UInt activeColumns[1] = {0};
      const vector<CellIdx> previousActiveCells =
        {4, 5, 6, 7}; // (there are more)
      vector<CellIdx> nonmatchingCells = {0, 3};
      vector<CellIdx> activeCells = {0, 1, 2, 3};

      Segment segment1 = tm.basalConnections.createSegment(nonmatchingCells[0]);
      tm.basalConnections.createSynapse(segment1, previousActiveCells[0], 0.5);
      Segment segment2 = tm.basalConnections.createSegment(nonmatchingCells[1]);
      tm.basalConnections.createSynapse(segment2, previousActiveCells[1], 0.5);

      tm.compute(4, previousActiveColumns, true);
      tm.compute(1, activeColumns, true);

      ASSERT_EQ(activeCells, tm.getActiveCells());

      EXPECT_EQ(3, tm.basalConnections.numSegments());
      EXPECT_EQ(1, tm.basalConnections.segmentsForCell(0).size());
      EXPECT_EQ(1, tm.basalConnections.segmentsForCell(3).size());
      EXPECT_EQ(1, tm.basalConnections.numSynapses(segment1));
      EXPECT_EQ(1, tm.basalConnections.numSynapses(segment2));

      Segment grownSegment;
      vector<Segment> segments = tm.basalConnections.segmentsForCell(1);
      if (segments.empty())
      {
        vector<Segment> segments2 = tm.basalConnections.segmentsForCell(2);
        EXPECT_FALSE(segments2.empty());
        grewOnCell2 = true;
        segments.insert(segments.end(), segments2.begin(), segments2.end());
      }
      else
      {
        grewOnCell1 = true;
      }

      ASSERT_EQ(1, segments.size());
      vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
      EXPECT_EQ(4, synapses.size());

      set<CellIdx> columnChecklist(previousActiveColumns, previousActiveColumns+4);

      for (Synapse synapse : synapses)
      {
        SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
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
  TEST(ExtendedTemporalMemoryTest, MaxNewSynapseCountOverflow)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    Segment segment = tm.basalConnections.createSegment(8);
    tm.basalConnections.createSynapse(segment, 0, 0.2);
    tm.basalConnections.createSynapse(segment, 1, 0.2);
    tm.basalConnections.createSynapse(segment, 2, 0.2);
    tm.basalConnections.createSynapse(segment, 3, 0.2);
    tm.basalConnections.createSynapse(segment, 4, 0.2);
    Synapse sampleSynapse = tm.basalConnections.createSynapse(segment, 5, 0.2);
    tm.basalConnections.createSynapse(segment, 6, 0.2);
    tm.basalConnections.createSynapse(segment, 7, 0.2);

    const UInt previousActiveColumns[4] = {0, 1, 3, 4};
    tm.compute(4, previousActiveColumns);

    ASSERT_EQ(1, tm.getMatchingBasalSegments().size());

    const UInt activeColumns[1] = {2};
    tm.compute(1, activeColumns);

    // Make sure the segment has learned.
    ASSERT_NEAR(0.3, tm.basalConnections.dataForSynapse(sampleSynapse).permanence,
                EPSILON);

    EXPECT_EQ(8, tm.basalConnections.numSynapses(segment));
  }

  /**
   * With learning disabled, generate some predicted active columns, predicted
   * inactive columns, and nonpredicted active columns. The connections should
   * not change.
   */
  TEST(ExtendedTemporalMemoryTest, ConnectionsNeverChangeWhenLearningDisabled)
  {
    ExtendedTemporalMemory tm(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
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
      tm.basalConnections.createSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[2], 0.5);

    Segment wrongMatchingSegment = tm.basalConnections.createSegment(43);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousInactiveCell, 0.5);

    Connections before = tm.basalConnections;

    tm.compute(1, previousActiveColumns, false);
    tm.compute(2, activeColumns, false);

    EXPECT_EQ(before, tm.basalConnections);
  }

  TEST(ExtendedTemporalMemoryTest, testColumnForCell1D)
  {
    ExtendedTemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    ASSERT_EQ(0, tm.columnForCell(0));
    ASSERT_EQ(0, tm.columnForCell(4));
    ASSERT_EQ(1, tm.columnForCell(5));
    ASSERT_EQ(2047, tm.columnForCell(10239));
  }

  TEST(ExtendedTemporalMemoryTest, testColumnForCell2D)
  {
    ExtendedTemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    ASSERT_EQ(0, tm.columnForCell(0));
    ASSERT_EQ(0, tm.columnForCell(3));
    ASSERT_EQ(1, tm.columnForCell(4));
    ASSERT_EQ(4095, tm.columnForCell(16383));
  }

  TEST(ExtendedTemporalMemoryTest, testColumnForCellInvalidCell)
  {
    ExtendedTemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    EXPECT_NO_THROW(tm.columnForCell(16383));
    EXPECT_THROW(tm.columnForCell(16384), std::exception);
    EXPECT_THROW(tm.columnForCell(-1), std::exception);
  }

  TEST(ExtendedTemporalMemoryTest, testNumberOfColumns)
  {
    ExtendedTemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    int numOfColumns = tm.numberOfColumns();
    ASSERT_EQ(numOfColumns, 64 * 64);
  }

  TEST(ExtendedTemporalMemoryTest, testNumberOfCells)
  {
    ExtendedTemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 32);

    Int numberOfCells = tm.numberOfCells();
    ASSERT_EQ(numberOfCells, 64 * 64 * 32);
  }

  TEST(ExtendedTemporalMemoryTest, testSaveLoad)
  {
    const char* filename = "ExtendedTemporalMemorySerialization.tmp";

    ExtendedTemporalMemory tm1(
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
      /*formInternalBasalConnections*/ true,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const UInt numActiveColumns = 1;
    const UInt previousActiveColumns[1] = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment activeSegment =
      tm1.basalConnections.createSegment(expectedActiveCells[0]);
    tm1.basalConnections.createSynapse(activeSegment, previousActiveCells[0],
                                       0.5);
    tm1.basalConnections.createSynapse(activeSegment, previousActiveCells[1],
                                       0.5);
    tm1.basalConnections.createSynapse(activeSegment, previousActiveCells[2],
                                       0.5);
    tm1.basalConnections.createSynapse(activeSegment, previousActiveCells[3],
                                       0.5);

    tm1.compute(numActiveColumns, previousActiveColumns, true);
    ASSERT_EQ(expectedActiveCells, tm1.getPredictiveCells());

    {
      ofstream outfile;
      outfile.open(filename, ios::binary);
      tm1.save(outfile);
      outfile.close();
    }

    ExtendedTemporalMemory tm2;

    {
      ifstream infile(filename, ios::binary);
      tm2.load(infile);
      infile.close();
    }

    check_tm_eq(tm1, tm2);

    int ret = ::remove(filename);
    ASSERT_EQ(0, ret) << "Failed to delete " << filename;
  }

  TEST(ExtendedTemporalMemoryTest, testWrite)
  {
    ExtendedTemporalMemory tm1, tm2;

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
    ASSERT_EQ(tm1.getActiveCells(), tm2.getActiveCells());
    ASSERT_EQ(tm1.getWinnerCells(), tm2.getWinnerCells());
    ASSERT_EQ(tm1.basalConnections, tm2.basalConnections);

    tm1.compute(sequence[3].size(), sequence[3].data());
    tm2.compute(sequence[3].size(), sequence[3].data());
    ASSERT_EQ(tm1.getActiveCells(), tm2.getActiveCells());

    ASSERT_EQ(tm1.getActiveBasalSegments(), tm2.getActiveBasalSegments());
    ASSERT_EQ(tm1.getMatchingBasalSegments(), tm2.getMatchingBasalSegments());

    ASSERT_EQ(tm1.getWinnerCells(), tm2.getWinnerCells());
    ASSERT_EQ(tm1.basalConnections, tm2.basalConnections);

    check_tm_eq(tm1, tm2);
  }
}
