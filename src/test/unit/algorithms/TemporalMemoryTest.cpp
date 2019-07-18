/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

/** @file
 * Implementation of unit tests for TemporalMemory
 */

#include <cstring>
#include <fstream>
#include <htm/utils/StlIo.hpp>
#include <htm/types/Types.hpp>
#include <htm/types/Sdr.hpp>
#include <htm/utils/Log.hpp>
#include <stdio.h>

#include "gtest/gtest.h"
#include <htm/algorithms/TemporalMemory.hpp>


namespace testing {

using namespace std;
using namespace htm;
#define EPSILON 0.0000001


TEST(TemporalMemoryTest, testInitInvalidParams) {
  TemporalMemory tm1;
  // Invalid columnDimensions
  EXPECT_ANY_THROW(tm1.initialize({}, 32));
  // Invalid cellsPerColumn
  EXPECT_ANY_THROW(tm1.initialize({2048}, 0));
  EXPECT_NO_THROW(tm1.initialize({2048}, 32));
}


/**
 * When a predicted column is activated, only the predicted cells in the
 * columns should be activated.
 */
TEST(TemporalMemoryTest, ActivateCorrectlyPredictiveCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> expectedActiveCells = {4};

  Segment activeSegment = tm.createSegment(expectedActiveCells[0]);
  tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.5f);

  tm.compute(previousActiveColumns, true);
  tm.activateDendrites();
  ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells().getSparse());
  tm.compute(activeColumns, true);

  EXPECT_EQ(expectedActiveCells, tm.getActiveCells());
}

/**
 * When an unpredicted column is activated, every cell in the column should
 * become active.
 */
TEST(TemporalMemoryTest, BurstUnpredictedColumns) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{0});
  const vector<CellIdx> burstingCells = {0, 1, 2, 3};

  tm.compute(activeColumns, true);

  EXPECT_EQ(burstingCells, tm.getActiveCells());
}

/**
 * When the TemporalMemory receives zero active columns, it should still
 * compute the active cells, winner cells, and predictive cells. All should be
 * empty.
 */
TEST(TemporalMemoryTest, ZeroActiveColumns) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);


  // Make some cells predictive.
  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> expectedActiveCells = {4};

  Segment segment = tm.createSegment(expectedActiveCells[0]);
  tm.connections.createSynapse(segment, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(segment, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(segment, previousActiveCells[2], 0.5f);
  tm.connections.createSynapse(segment, previousActiveCells[3], 0.5f);
  tm.compute(previousActiveColumns, true);
  ASSERT_FALSE(tm.getActiveCells().empty());
  ASSERT_FALSE(tm.getWinnerCells().empty());
  tm.activateDendrites();
  ASSERT_FALSE(tm.getPredictiveCells().getSum() == 0);

  SDR empty({32});
  empty.setSparse(SDR_sparse_t{});
  EXPECT_NO_THROW(tm.compute(empty, true)) << "failed with empty compute";
  EXPECT_TRUE(tm.getActiveCells().empty());
  EXPECT_TRUE(tm.getWinnerCells().empty());
  tm.activateDendrites();
  EXPECT_TRUE(tm.getPredictiveCells().getSum() == 0);
}

/**
 * All predicted active cells are winner cells, even when learning is
 * disabled.
 */
TEST(TemporalMemoryTest, PredictedActiveCellsAreAlwaysWinners) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> expectedWinnerCells = {4, 6};

  Segment activeSegment1 = tm.createSegment(expectedWinnerCells[0]);
  tm.connections.createSynapse(activeSegment1, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment1, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment1, previousActiveCells[2], 0.5f);

  Segment activeSegment2 = tm.createSegment(expectedWinnerCells[1]);
  tm.connections.createSynapse(activeSegment2, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment2, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment2, previousActiveCells[2], 0.5f);

  tm.compute(previousActiveColumns, false);
  tm.compute(activeColumns, false);

  EXPECT_EQ(expectedWinnerCells, tm.getWinnerCells());
}

/**
 * One cell in each bursting column is a winner cell, even when learning is
 * disabled.
 */
TEST(TemporalMemoryTest, ChooseOneWinnerCellInBurstingColumn) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{0});
  const set<CellIdx> burstingCells = {0, 1, 2, 3};

  tm.compute(activeColumns, false);

  vector<CellIdx> winnerCells = tm.getWinnerCells();
  ASSERT_EQ(1ul, winnerCells.size());
  EXPECT_TRUE(burstingCells.find(winnerCells[0]) != burstingCells.end());
}

/**
 * Active segments on predicted active cells should be reinforced. Active
 * synapses should be reinforced, inactive synapses should be punished.
 */
TEST(TemporalMemoryTest, ReinforceCorrectlyActiveSegments) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.08f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});  
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});  
  const vector<CellIdx> activeCells = {5};
  const CellIdx activeCell = 5;

  Segment activeSegment = tm.createSegment(activeCell);
  Synapse activeSynapse1 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  Synapse activeSynapse2 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  Synapse activeSynapse3 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);
  Synapse inactiveSynapse =
      tm.connections.createSynapse(activeSegment, 81, 0.5f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

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
TEST(TemporalMemoryTest, ReinforceSelectedMatchingSegmentInBurstingColumn) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.08f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> burstingCells = {4, 5, 6, 7};

  Segment selectedMatchingSegment = tm.createSegment(burstingCells[0]);
  Synapse activeSynapse1 = tm.connections.createSynapse(
      selectedMatchingSegment, previousActiveCells[0], 0.3f);
  Synapse activeSynapse2 = tm.connections.createSynapse(
      selectedMatchingSegment, previousActiveCells[1], 0.3f);
  Synapse activeSynapse3 = tm.connections.createSynapse(
      selectedMatchingSegment, previousActiveCells[2], 0.3f);
  Synapse inactiveSynapse =
      tm.connections.createSynapse(selectedMatchingSegment, 81, 0.3f);

  // Add some competition.
  Segment otherMatchingSegment = tm.createSegment(burstingCells[1]);
  tm.connections.createSynapse(otherMatchingSegment, previousActiveCells[0],
                               0.3f);
  tm.connections.createSynapse(otherMatchingSegment, previousActiveCells[1],
                               0.3f);
  tm.connections.createSynapse(otherMatchingSegment, 81, 0.3f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_NEAR(0.4f, tm.connections.dataForSynapse(activeSynapse1).permanence,
              EPSILON);
  EXPECT_NEAR(0.4f, tm.connections.dataForSynapse(activeSynapse2).permanence,
              EPSILON);
  EXPECT_NEAR(0.4f, tm.connections.dataForSynapse(activeSynapse3).permanence,
              EPSILON);
  EXPECT_NEAR(0.22f, tm.connections.dataForSynapse(inactiveSynapse).permanence,
              EPSILON);
}

/**
 * When a column bursts, don't reward or punish matching-but-not-selected
 * segments.
 */
TEST(TemporalMemoryTest,
     NoChangeToNonselectedMatchingSegmentsInBurstingColumn) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.08f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> burstingCells = {4, 5, 6, 7};

  Segment selectedMatchingSegment = tm.createSegment(burstingCells[0]);
  tm.connections.createSynapse(selectedMatchingSegment, previousActiveCells[0],
                               0.3f);
  tm.connections.createSynapse(selectedMatchingSegment, previousActiveCells[1],
                               0.3f);
  tm.connections.createSynapse(selectedMatchingSegment, previousActiveCells[2],
                               0.3f);
  tm.connections.createSynapse(selectedMatchingSegment, 81, 0.3f);

  Segment otherMatchingSegment = tm.createSegment(burstingCells[1]);
  Synapse activeSynapse1 = tm.connections.createSynapse(
      otherMatchingSegment, previousActiveCells[0], 0.3f);
  Synapse activeSynapse2 = tm.connections.createSynapse(
      otherMatchingSegment, previousActiveCells[1], 0.3f);
  Synapse inactiveSynapse =
      tm.connections.createSynapse(otherMatchingSegment, 81, 0.3f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(activeSynapse1).permanence,
              EPSILON);
  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(activeSynapse2).permanence,
              EPSILON);
  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(inactiveSynapse).permanence,
              EPSILON);
}

/**
 * When a predicted column is activated, don't reward or punish
 * matching-but-not-active segments anywhere in the column.
 */
TEST(TemporalMemoryTest, NoChangeToMatchingSegmentsInPredictedActiveColumn) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const vector<CellIdx> expectedActiveCells = {4};
  const vector<CellIdx> otherBurstingCells = {5, 6, 7};

  Segment activeSegment = tm.createSegment(expectedActiveCells[0]);
  tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.5f);

  Segment matchingSegmentOnSameCell = tm.createSegment(expectedActiveCells[0]);
  Synapse synapse1 = tm.connections.createSynapse(matchingSegmentOnSameCell,
                                                  previousActiveCells[0], 0.3f);
  Synapse synapse2 = tm.connections.createSynapse(matchingSegmentOnSameCell,
                                                  previousActiveCells[1], 0.3f);

  Segment matchingSegmentOnOtherCell = tm.createSegment(otherBurstingCells[0]);
  Synapse synapse3 = tm.connections.createSynapse(matchingSegmentOnOtherCell,
                                                  previousActiveCells[0], 0.3f);
  Synapse synapse4 = tm.connections.createSynapse(matchingSegmentOnOtherCell,
                                                  previousActiveCells[1], 0.3f);

  tm.compute(previousActiveColumns, true);
  tm.activateDendrites();
  ASSERT_EQ(expectedActiveCells, tm.getPredictiveCells().getSparse());
  tm.compute(activeColumns, true);

  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(synapse1).permanence, EPSILON);
  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(synapse2).permanence, EPSILON);
  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(synapse3).permanence, EPSILON);
  EXPECT_NEAR(0.3f, tm.connections.dataForSynapse(synapse4).permanence, EPSILON);
}

/**
 * When growing a new segment, if there are no previous winner cells, don't
 * even grow the segment. It will never match.
 */
TEST(TemporalMemoryTest, NoNewSegmentIfNotEnoughWinnerCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 2,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR zeroColumns({32});
  zeroColumns.setSparse(SDR_sparse_t{});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});

  tm.compute(zeroColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_EQ(0ul, tm.connections.numSegments());
}

/**
 * When growing a new segment, if the number of previous winner cells is above
 * maxNewSynapseCount, grow maxNewSynapseCount synapses.
 */
TEST(TemporalMemoryTest, NewSegmentAddSynapsesToSubsetOfWinnerCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 2,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1,2});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{4});

  tm.compute(previousActiveColumns);

  auto prevWinnerCells = tm.getWinnerCells();
  ASSERT_EQ(3ul, prevWinnerCells.size());

  tm.compute(activeColumns);

  auto winnerCells = tm.getWinnerCells();
  ASSERT_EQ(1ul, winnerCells.size());
  vector<Segment> segments = tm.connections.segmentsForCell(winnerCells[0]);
  ASSERT_EQ(1ul, segments.size());
  vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
  ASSERT_EQ(2ul, synapses.size());
  for (Synapse synapse : synapses) {
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
TEST(TemporalMemoryTest, NewSegmentAddSynapsesToAllWinnerCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1,2});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{4});

  tm.compute(previousActiveColumns);

  vector<CellIdx> prevWinnerCells = tm.getWinnerCells();
  ASSERT_EQ(3ul, prevWinnerCells.size());

  tm.compute(activeColumns);

  vector<CellIdx> winnerCells = tm.getWinnerCells();
  ASSERT_EQ(1ul, winnerCells.size());
  vector<Segment> segments = tm.connections.segmentsForCell(winnerCells[0]);
  ASSERT_EQ(1ul, segments.size());
  vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
  ASSERT_EQ(3ul, synapses.size());

  vector<CellIdx> presynapticCells;
  for (Synapse synapse : synapses) {
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
TEST(TemporalMemoryTest, MatchingSegmentAddSynapsesToSubsetOfWinnerCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  // Use 1 cell per column so that we have easy control over the winner cells.
  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1,2,3});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{4});
  const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3};

  Segment matchingSegment = tm.createSegment(4);
  tm.connections.createSynapse(matchingSegment, 0, 0.5f);

  tm.compute(previousActiveColumns);

  ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

  tm.compute(activeColumns);

  vector<Synapse> synapses = tm.connections.synapsesForSegment(matchingSegment);
  ASSERT_EQ(3ul, synapses.size());
  for (SynapseIdx i = 1; i < synapses.size(); i++) {
    SynapseData synapseData = tm.connections.dataForSynapse(synapses[i]);
    EXPECT_NEAR(0.21f, synapseData.permanence, EPSILON);
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
TEST(TemporalMemoryTest, MatchingSegmentAddSynapsesToAllWinnerCells) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  // Use 1 cell per column so that we have easy control over the winner cells.
  const vector<CellIdx> prevWinnerCells = {0, 1};
  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{4});

  Segment matchingSegment = tm.createSegment(4);
  tm.connections.createSynapse(matchingSegment, 0, 0.5f);

  tm.compute(previousActiveColumns);

  ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

  tm.compute(activeColumns);

  vector<Synapse> synapses = tm.connections.synapsesForSegment(matchingSegment);
  ASSERT_EQ(2ul, synapses.size());

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
TEST(TemporalMemoryTest, ActiveSegmentGrowSynapsesAccordingToPotentialOverlap) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 2,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  // Use 1 cell per column so that we have easy control over the winner cells.
  const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3, 4};
  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1,2,3,4});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{5});

  Segment activeSegment = tm.createSegment(5);
  tm.connections.createSynapse(activeSegment, 0, 0.5f);
  tm.connections.createSynapse(activeSegment, 1, 0.5f);
  tm.connections.createSynapse(activeSegment, 2, 0.2f);

  tm.compute(previousActiveColumns);

  ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

  tm.compute(activeColumns);

  vector<Synapse> synapses = tm.connections.synapsesForSegment(activeSegment);

  ASSERT_EQ(4ul, synapses.size());

  SynapseData synapseData = tm.connections.dataForSynapse(synapses[3]);
  EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
  EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[3] ||
              synapseData.presynapticCell == prevWinnerCells[4]);
}

/**
 * When a synapse is punished for contributing to a wrong prediction, if its
 * permanence falls to 0 it should be destroyed.
 */
TEST(TemporalMemoryTest, DestroyWeakSynapseOnWrongPrediction) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{2});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const CellIdx expectedActiveCell = 5;

  Segment activeSegment = tm.createSegment(expectedActiveCell);
  tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);

  // Weak synapse.
  tm.connections.createSynapse(activeSegment, previousActiveCells[3], 0.015f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_EQ(3ul, tm.connections.numSynapses(activeSegment));
}

/**
 * When a synapse is punished for not contributing to a right prediction, if
 * its permanence falls to 0 it should be destroyed.
 */
TEST(TemporalMemoryTest, DestroyWeakSynapseOnActiveReinforce) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const CellIdx activeCell = 5;

  Segment activeSegment = tm.createSegment(activeCell);
  tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);

  // Weak inactive synapse.
  tm.connections.createSynapse(activeSegment, 81, 0.09f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_EQ(3ul, tm.connections.numSynapses(activeSegment));
}

/**
 * When a segment adds synapses and it runs over maxSynapsesPerSegment, it
 * should make room by destroying synapses with the lowest permanence.
 */
TEST(TemporalMemoryTest, RecycleWeakestSynapseToMakeRoomForNewSynapse) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 1,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02f,
      /*permanenceDecrement*/ 0.02f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 255,
      /*maxSynapsesPerSegment*/ 4);

  // Use 1 cell per column so that we have easy control over the winner cells.
  const vector<CellIdx> prevWinnerCells = {1, 2, 3};
  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{1,2,3});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{4});

  Segment matchingSegment = tm.createSegment(4);

  // Create a weak synapse. Make sure it's not so weak that
  // permanenceDecrement destroys it.
  tm.connections.createSynapse(matchingSegment, 0, 0.11f);

  // Create a synapse that will match.
  tm.connections.createSynapse(matchingSegment, 1, 0.20f);

  // Create a synapse with a high permanence.
  tm.connections.createSynapse(matchingSegment, 31, 0.6f);

  // Activate a synapse on the segment, making it "matching".
  tm.compute(previousActiveColumns);

  ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

  // Now mark the segment as "correct" by activating its cell.
  tm.compute(activeColumns);

  // There should now be 3 synapses, and none of them should be to cell 0.
  const vector<Synapse> &synapses =
      tm.connections.synapsesForSegment(matchingSegment);
  ASSERT_EQ(4ul, synapses.size());

  std::set<CellIdx> presynapticCells;
  for (Synapse synapse : synapses) {
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
TEST(TemporalMemoryTest,
     RecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02f,
      /*permanenceDecrement*/ 0.02f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2);

  SDR previousActiveColumns1({32});
  previousActiveColumns1.setSparse(SDR_sparse_t{0,1,2});
  SDR previousActiveColumns2({32});
  previousActiveColumns2.setSparse(SDR_sparse_t{3,4,5});
  SDR previousActiveColumns3({32});
  previousActiveColumns3.setSparse(SDR_sparse_t{6,7,8});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{9});

  tm.compute(previousActiveColumns1);
  tm.compute(activeColumns);

  ASSERT_EQ(1ul, tm.connections.numSegments(9));
  Segment oldestSegment = tm.connections.segmentsForCell(9)[0];

  tm.reset();
  tm.compute(previousActiveColumns2);
  tm.compute(activeColumns);

  ASSERT_EQ(2ul, tm.connections.numSegments(9));

  set<CellIdx> oldPresynaptic;
  for (Synapse synapse : tm.connections.synapsesForSegment(oldestSegment)) {
    oldPresynaptic.insert(
        tm.connections.dataForSynapse(synapse).presynapticCell);
  }

  tm.reset();
  tm.compute(previousActiveColumns3);
  tm.compute(activeColumns);

  ASSERT_EQ(2ul, tm.connections.numSegments(9));

  // Verify none of the segments are connected to the cells the old segment
  // was connected to.

  for (Segment segment : tm.connections.segmentsForCell(9)) {
    set<CellIdx> newPresynaptic;
    for (Synapse synapse : tm.connections.synapsesForSegment(segment)) {
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
TEST(TemporalMemoryTest, DestroySegmentsWithTooFewSynapsesToBeMatching) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{2});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const CellIdx expectedActiveCell = 5;

  Segment matchingSegment = tm.createSegment(expectedActiveCell);
  tm.connections.createSynapse(matchingSegment, previousActiveCells[0], 0.015f);
  tm.connections.createSynapse(matchingSegment, previousActiveCells[1], 0.015f);
  tm.connections.createSynapse(matchingSegment, previousActiveCells[2], 0.015f);
  tm.connections.createSynapse(matchingSegment, previousActiveCells[3], 0.015f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_EQ(0ul, tm.connections.numSegments(expectedActiveCell));
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
TEST(TemporalMemoryTest, PunishMatchingSegmentsInInactiveColumns) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const CellIdx previousInactiveCell = 81;

  Segment activeSegment = tm.createSegment(42);
  Synapse activeSynapse1 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[0], 0.5f);
  Synapse activeSynapse2 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[1], 0.5f);
  Synapse activeSynapse3 =
      tm.connections.createSynapse(activeSegment, previousActiveCells[2], 0.5f);
  Synapse inactiveSynapse1 =
      tm.connections.createSynapse(activeSegment, previousInactiveCell, 0.5f);

  Segment matchingSegment = tm.createSegment(43);
  Synapse activeSynapse4 = tm.connections.createSynapse(
      matchingSegment, previousActiveCells[0], 0.5f);
  Synapse activeSynapse5 = tm.connections.createSynapse(
      matchingSegment, previousActiveCells[1], 0.5f);
  Synapse inactiveSynapse2 =
      tm.connections.createSynapse(matchingSegment, previousInactiveCell, 0.5f);

  tm.compute(previousActiveColumns, true);
  tm.compute(activeColumns, true);

  EXPECT_NEAR(0.48f, tm.connections.dataForSynapse(activeSynapse1).permanence,
              EPSILON);
  EXPECT_NEAR(0.48f, tm.connections.dataForSynapse(activeSynapse2).permanence,
              EPSILON);
  EXPECT_NEAR(0.48f, tm.connections.dataForSynapse(activeSynapse3).permanence,
              EPSILON);
  EXPECT_NEAR(0.48f, tm.connections.dataForSynapse(activeSynapse4).permanence,
              EPSILON);
  EXPECT_NEAR(0.48f, tm.connections.dataForSynapse(activeSynapse5).permanence,
              EPSILON);
  EXPECT_NEAR(0.50f, tm.connections.dataForSynapse(inactiveSynapse1).permanence,
              EPSILON);
  EXPECT_NEAR(0.50f, tm.connections.dataForSynapse(inactiveSynapse2).permanence,
              EPSILON);
}

/**
 * In a bursting column with no matching segments, a segment should be added
 * to the cell with the fewest segments. When there's a tie, choose randomly.
 */
TEST(TemporalMemoryTest, AddSegmentToCellWithFewestSegments) {
  bool grewOnCell1 = false;
  bool grewOnCell2 = false;
  for (UInt seed = 0; seed < 100; seed++) {
    TemporalMemory tm(
        /*columnDimensions*/ {32},
        /*cellsPerColumn*/ 4,
        /*activationThreshold*/ 3,
        /*initialPermanence*/ 0.2f,
        /*connectedPermanence*/ 0.50f,
        /*minThreshold*/ 2,
        /*maxNewSynapseCount*/ 4,
        /*permanenceIncrement*/ 0.10f,
        /*permanenceDecrement*/ 0.10f,
        /*predictedSegmentDecrement*/ 0.02f,
        /*seed*/ seed);

    // enough for 4 winner cells
    SDR previousActiveColumns({32});
    previousActiveColumns.setSparse(SDR_sparse_t{1,2,3,4});
    SDR activeColumns({32});
    activeColumns.setSparse(SDR_sparse_t{0});
    const vector<CellIdx> previousActiveCells = {4, 5, 6,7}; // (there are more)
    vector<CellIdx> nonmatchingCells = {0, 3};
    vector<CellIdx> activeCells = {0, 1, 2, 3};

    Segment segment1 = tm.createSegment(nonmatchingCells[0]);
    tm.connections.createSynapse(segment1, previousActiveCells[0], 0.5f);
    Segment segment2 = tm.createSegment(nonmatchingCells[1]);
    tm.connections.createSynapse(segment2, previousActiveCells[1], 0.5f);

    tm.compute(previousActiveColumns, true);
    tm.compute(activeColumns, true);

    ASSERT_EQ(activeCells, tm.getActiveCells());

    EXPECT_EQ(3ul, tm.connections.numSegments());
    EXPECT_EQ(1ul, tm.connections.segmentsForCell(0).size());
    EXPECT_EQ(1ul, tm.connections.segmentsForCell(3).size());
    EXPECT_EQ(1ul, tm.connections.numSynapses(segment1));
    EXPECT_EQ(1ul, tm.connections.numSynapses(segment2));

    vector<Segment> segments = tm.connections.segmentsForCell(1);
    if (segments.empty()) {
      vector<Segment> segments2 = tm.connections.segmentsForCell(2);
      EXPECT_FALSE(segments2.empty());
      grewOnCell2 = true;
      segments.insert(segments.end(), segments2.begin(), segments2.end());
    } else {
      grewOnCell1 = true;
    }

    ASSERT_EQ(1ul, segments.size());
    vector<Synapse> synapses = tm.connections.synapsesForSegment(segments[0]);
    EXPECT_EQ(4ul, synapses.size());

    set<CellIdx> columnChecklist(previousActiveColumns.getSparse().data(),
                                 previousActiveColumns.getSparse().data() + 4);

    for (Synapse synapse : synapses) {
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
TEST(TemporalMemoryTest, MaxNewSynapseCountOverflow) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  Segment segment = tm.createSegment(8);
  tm.connections.createSynapse(segment, 0, 0.2f);
  tm.connections.createSynapse(segment, 1, 0.2f);
  tm.connections.createSynapse(segment, 2, 0.2f);
  tm.connections.createSynapse(segment, 3, 0.2f);
  tm.connections.createSynapse(segment, 4, 0.2f);
  Synapse sampleSynapse = tm.connections.createSynapse(segment, 5, 0.2f);
  tm.connections.createSynapse(segment, 6, 0.2f);
  tm.connections.createSynapse(segment, 7, 0.2f);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0,1,3,4});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{2});

  tm.compute(previousActiveColumns);
  tm.activateDendrites();

  ASSERT_EQ(1ul, tm.getMatchingSegments().size());

  tm.compute(activeColumns);

  // Make sure the segment has learned.
  ASSERT_NEAR(0.3, tm.connections.dataForSynapse(sampleSynapse).permanence, EPSILON);
  EXPECT_EQ(8ul, tm.connections.numSynapses(segment));
}

/**
 * With learning disabled, generate some predicted active columns, predicted
 * inactive columns, and nonpredicted active columns. The connections should
 * not change.
 */
TEST(TemporalMemoryTest, ConnectionsNeverChangeWhenLearningDisabled) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 4,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.02f,
      /*seed*/ 42);

  SDR previousActiveColumns({32});
  previousActiveColumns.setSparse(SDR_sparse_t{0});
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1 /*predicted*/, 2 /*bursting*/});
  const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
  const CellIdx previousInactiveCell = 81;
  const vector<CellIdx> expectedActiveCells = {4};

  Segment correctActiveSegment = tm.createSegment(expectedActiveCells[0]);
  tm.connections.createSynapse(correctActiveSegment, previousActiveCells[0],
                               0.5f);
  tm.connections.createSynapse(correctActiveSegment, previousActiveCells[1],
                               0.5f);
  tm.connections.createSynapse(correctActiveSegment, previousActiveCells[2],
                               0.5f);

  Segment wrongMatchingSegment = tm.createSegment(43);
  tm.connections.createSynapse(wrongMatchingSegment, previousActiveCells[0],
                               0.5f);
  tm.connections.createSynapse(wrongMatchingSegment, previousActiveCells[1],
                               0.5f);
  tm.connections.createSynapse(wrongMatchingSegment, previousInactiveCell, 0.5f);

  const Connections before = tm.connections;

  tm.compute(previousActiveColumns, false);
  tm.compute(activeColumns, false);

  EXPECT_EQ(before, tm.connections);
}

/**
 * Destroy some segments then verify that the maxSegmentsPerCell is still
 * correctly applied.
 */
TEST(TemporalMemoryTest, DestroySegmentsThenReachLimit) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02f,
      /*permanenceDecrement*/ 0.02f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2);

  {
    Segment segment1 = tm.createSegment(11);
    Segment segment2 = tm.createSegment(11);
    ASSERT_EQ(2ul, tm.connections.numSegments());
    tm.connections.destroySegment(segment1);
    tm.connections.destroySegment(segment2);
    ASSERT_EQ(0ul, tm.connections.numSegments());
  }

  {
    tm.createSegment(11);
    EXPECT_EQ(1ul, tm.connections.numSegments());
    tm.createSegment(11);
    EXPECT_EQ(2ul, tm.connections.numSegments());
    tm.createSegment(11);
    EXPECT_EQ(2ul, tm.connections.numSegments()) << "Created 3 segments, but limit is 2, so this should be 2!";
    EXPECT_EQ(2ul, tm.connections.numSegments(11));
  }
}

/**
 * Creates many segments on a cell, until hits segment limit. Then creates
 * another segment, and checks that it destroyed the least recently used
 * segment and created a new one in its place.
 */
TEST(TemporalMemoryTest, CreateSegmentDestroyOld) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02f,
      /*permanenceDecrement*/ 0.02f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2);

  Segment segment1 = tm.createSegment(12);

  tm.connections.createSynapse(segment1, 1, 0.5f);
  tm.connections.createSynapse(segment1, 2, 0.5f);
  tm.connections.createSynapse(segment1, 3, 0.5f);

  // Let some time pass.
  SDR empty({32});
  tm.compute(empty);
  tm.compute(empty);
  tm.compute(empty);

  // Create a segment with 1 synapse.
  Segment segment2 = tm.createSegment(12);
  tm.connections.createSynapse(segment2, 1, 0.5f);

  tm.compute(empty);

  // Give the first segment some activity.
  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1,2,3});

  tm.compute(activeColumns);
  tm.activateDendrites();
  ASSERT_EQ( tm.getActiveSegments(), vector<Segment>({ segment1 }) );

  tm.compute(empty);

  // Create a new segment with two synapses.
  Segment segment3 = tm.createSegment(12);
  tm.connections.createSynapse(segment3, 1, 0.5);
  tm.connections.createSynapse(segment3, 2, 0.5);

  vector<Segment> segments = tm.connections.segmentsForCell(12);
  ASSERT_EQ(2ul, segments.size());

  // Verify first segment is still there with the same synapses.
  vector<Synapse> synapses1 = tm.connections.synapsesForSegment(segment1);
  ASSERT_EQ(3ul, synapses1.size());
  ASSERT_EQ(1ul, tm.connections.dataForSynapse(synapses1[0]).presynapticCell);
  ASSERT_EQ(2ul, tm.connections.dataForSynapse(synapses1[1]).presynapticCell);
  ASSERT_EQ(3ul, tm.connections.dataForSynapse(synapses1[2]).presynapticCell);

  // Verify second segment has been replaced.
  ASSERT_EQ(2ul, tm.connections.numSynapses(segments[1]));
}

/**
 * Hit the maxSegmentsPerCell threshold multiple times. Make sure it works
 * more than once.
 */
TEST(TemporalMemoryTest, ReachSegmentLimitMultipleTimes) {
  TemporalMemory tm(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.02f,
      /*permanenceDecrement*/ 0.02f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2);

  tm.createSegment(10);
  ASSERT_EQ(1ul, tm.connections.numSegments());
  tm.createSegment(10);
  ASSERT_EQ(2ul, tm.connections.numSegments());
  tm.createSegment(10);
  ASSERT_EQ(2ul, tm.connections.numSegments());
  tm.createSegment(10);
  EXPECT_EQ(2ul, tm.connections.numSegments());
}

TEST(TemporalMemoryTest, testColumnForCell1D) {
  TemporalMemory tm;
  tm.initialize(vector<UInt>{2048}, 5);

  ASSERT_EQ(0ul, tm.columnForCell(0));
  ASSERT_EQ(0ul, tm.columnForCell(4));
  ASSERT_EQ(1ul, tm.columnForCell(5));
  ASSERT_EQ(2047ul, tm.columnForCell(10239));
}

TEST(TemporalMemoryTest, testColumnForCell2D) {
  TemporalMemory tm;
  tm.initialize(vector<UInt>{64, 64}, 4);

  ASSERT_EQ(0ul, tm.columnForCell(0));
  ASSERT_EQ(0ul, tm.columnForCell(3));
  ASSERT_EQ(1ul, tm.columnForCell(4));
  ASSERT_EQ(4095ul, tm.columnForCell(16383));
}

TEST(TemporalMemoryTest, testColumnForCellInvalidCell) {
  TemporalMemory tm;
  tm.initialize(vector<UInt>{64, 64}, 4);

  EXPECT_NO_THROW(tm.columnForCell(16383));
  //does not throw in Release build (as NTA_ASSERT)
#ifndef NDEBUG
  EXPECT_THROW(tm.columnForCell(16384), std::exception);
  EXPECT_THROW(tm.columnForCell(-1), std::exception);
#endif
}

TEST(TemporalMemoryTest, testCellsToColumns)
{
  TemporalMemory tm;
  tm.initialize(vector<UInt>{3}, 3); // TM 3 cols x 3 cells per col

  auto correctDims = tm.getColumnDimensions();
  correctDims.push_back(static_cast<UInt>(tm.getCellsPerColumn()));
  SDR v1(correctDims);
  v1.setSparse(SDR_sparse_t{4,5,8});
  const SDR_sparse_t expected {1u, 2u};

  auto res = tm.cellsToColumns(v1);
  ASSERT_EQ(res.getSparse(), expected);

  v1.setSparse(SDR_sparse_t{}); // empty sparse array
  res = tm.cellsToColumns(v1);
  EXPECT_TRUE(res.getSparse().empty());

  SDR larger({10,10, 3});
  EXPECT_ANY_THROW(tm.cellsToColumns(larger));

  SDR wrongDims({3*3}); //matches numberOfCells, but dimensions are incorrect
  EXPECT_ANY_THROW(tm.cellsToColumns(wrongDims));
};


TEST(TemporalMemoryTest, testNumberOfColumns) {
  TemporalMemory tm;
  tm.initialize(vector<UInt>{64, 64}, 32);

  size_t numOfColumns = tm.numberOfColumns();
  ASSERT_EQ(numOfColumns, 64ull * 64ull);
}

TEST(TemporalMemoryTest, testNumberOfCells) {
  TemporalMemory tm;
  tm.initialize(vector<UInt>{64, 64}, 32);

  size_t numberOfCells = tm.numberOfCells();
  ASSERT_EQ(numberOfCells, 64ull * 64ull * 32ull);
}

void serializationTestPrepare(TemporalMemory &tm) {
  // Create an active segment and a two matching segments.
  // Destroy a few to exercise the code.
  Segment destroyMe1 = tm.createSegment(4);
  tm.connections.destroySegment(destroyMe1);

  Segment activeSegment = tm.createSegment(4);
  tm.connections.createSynapse(activeSegment, 0, 0.5f);
  tm.connections.createSynapse(activeSegment, 1, 0.5f);
  Synapse destroyMe2 = tm.connections.createSynapse(activeSegment, 42, 0.5f);
  tm.connections.destroySynapse(destroyMe2);
  tm.connections.createSynapse(activeSegment, 2, 0.5f);
  tm.connections.createSynapse(activeSegment, 3, 0.5f);

  Segment matchingSegment1 = tm.createSegment(8);
  tm.connections.createSynapse(matchingSegment1, 0, 0.4f);
  tm.connections.createSynapse(matchingSegment1, 1, 0.4f);
  tm.connections.createSynapse(matchingSegment1, 2, 0.4f);

  Segment matchingSegment2 = tm.createSegment(9);
  tm.connections.createSynapse(matchingSegment2, 0, 0.4f);
  tm.connections.createSynapse(matchingSegment2, 1, 0.4f);
  tm.connections.createSynapse(matchingSegment2, 2, 0.4f);
  tm.connections.createSynapse(matchingSegment2, 3, 0.4f);

  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{0});
  tm.compute(activeColumns);
  tm.activateDendrites();

  ASSERT_EQ(1ul, tm.getActiveSegments().size());
  ASSERT_EQ(3ul, tm.getMatchingSegments().size());
}

void serializationTestVerify(TemporalMemory &tm) {
  // Activate 3 columns. One has an active segment, one has two
  // matching segments, and one has none. One column should be
  // predicted, the others should burst, there should be four
  // segments total, and they should have the correct permanences
  // and synapse counts.

  const vector<UInt> prevWinnerCells = tm.getWinnerCells();
  ASSERT_EQ(1ul, prevWinnerCells.size());

  SDR activeColumns({32});
  activeColumns.setSparse(SDR_sparse_t{1,2,3});
  tm.compute(activeColumns);

  // Verify the correct cells were activated.
  EXPECT_EQ((vector<UInt>{4, 8, 9, 10, 11, 12, 13, 14, 15}),
            tm.getActiveCells());
  const vector<UInt> winnerCells = tm.getWinnerCells();
  ASSERT_EQ(3ul, winnerCells.size());
  EXPECT_EQ(4ul, winnerCells[0]);
  EXPECT_EQ(9ul, winnerCells[1]);

  EXPECT_EQ(4ul, tm.connections.numSegments());

  // Verify the active segment learned.
  ASSERT_EQ(1ul, tm.connections.numSegments(4));
  Segment activeSegment = tm.connections.segmentsForCell(4)[0];
  const vector<Synapse> syns1 =
      tm.connections.synapsesForSegment(activeSegment);
  ASSERT_EQ(4ul, syns1.size());
  EXPECT_EQ(0ul, tm.connections.dataForSynapse(syns1[0]).presynapticCell);
  EXPECT_NEAR(0.6, tm.connections.dataForSynapse(syns1[0]).permanence, EPSILON);
  EXPECT_EQ(1ul, tm.connections.dataForSynapse(syns1[1]).presynapticCell);
  EXPECT_NEAR(0.6, tm.connections.dataForSynapse(syns1[1]).permanence, EPSILON);
  EXPECT_EQ(2ul, tm.connections.dataForSynapse(syns1[2]).presynapticCell);
  EXPECT_NEAR(0.6, tm.connections.dataForSynapse(syns1[2]).permanence, EPSILON);
  EXPECT_EQ(3ul, tm.connections.dataForSynapse(syns1[3]).presynapticCell);
  EXPECT_NEAR(0.6, tm.connections.dataForSynapse(syns1[3]).permanence, EPSILON);

  // Verify the non-best matching segment is unchanged.
  ASSERT_EQ(1ul, tm.connections.numSegments(8));
  Segment matchingSegment1 = tm.connections.segmentsForCell(8)[0];
  const vector<Synapse> syns2 =
      tm.connections.synapsesForSegment(matchingSegment1);
  ASSERT_EQ(3ul, syns2.size());
  EXPECT_EQ(0ul, tm.connections.dataForSynapse(syns2[0]).presynapticCell);
  EXPECT_NEAR(0.4, tm.connections.dataForSynapse(syns2[0]).permanence, EPSILON);
  EXPECT_EQ(1ul, tm.connections.dataForSynapse(syns2[1]).presynapticCell);
  EXPECT_NEAR(0.4, tm.connections.dataForSynapse(syns2[1]).permanence, EPSILON);
  EXPECT_EQ(2ul, tm.connections.dataForSynapse(syns2[2]).presynapticCell);
  EXPECT_NEAR(0.4, tm.connections.dataForSynapse(syns2[2]).permanence, EPSILON);

  // Verify the best matching segment learned.
  ASSERT_EQ(1ul, tm.connections.numSegments(9));
  Segment matchingSegment2 = tm.connections.segmentsForCell(9)[0];
  const vector<Synapse> syns3 =
      tm.connections.synapsesForSegment(matchingSegment2);
  ASSERT_EQ(4ul, syns3.size());
  EXPECT_EQ(0ul, tm.connections.dataForSynapse(syns3[0]).presynapticCell);
  EXPECT_NEAR(0.5, tm.connections.dataForSynapse(syns3[0]).permanence, EPSILON);
  EXPECT_EQ(1ul, tm.connections.dataForSynapse(syns3[1]).presynapticCell);
  EXPECT_NEAR(0.5, tm.connections.dataForSynapse(syns3[1]).permanence, EPSILON);
  EXPECT_EQ(2ul, tm.connections.dataForSynapse(syns3[2]).presynapticCell);
  EXPECT_NEAR(0.5, tm.connections.dataForSynapse(syns3[2]).permanence, EPSILON);
  EXPECT_EQ(3ul, tm.connections.dataForSynapse(syns3[3]).presynapticCell);
  EXPECT_NEAR(0.5, tm.connections.dataForSynapse(syns3[3]).permanence, EPSILON);

  // Verify the winner cell in the last column grew a segment.
  const UInt winnerCell = winnerCells[2];
  EXPECT_GE(winnerCell, 12u);
  EXPECT_LT(winnerCell, 16u);
  ASSERT_EQ(1ul, tm.connections.numSegments(winnerCell));
  Segment newSegment = tm.connections.segmentsForCell(winnerCell)[0];
  const vector<Synapse> syns4 = tm.connections.synapsesForSegment(newSegment);
  ASSERT_EQ(1ul, syns4.size());
  EXPECT_EQ(prevWinnerCells[0],
            tm.connections.dataForSynapse(syns4[0]).presynapticCell);
  EXPECT_NEAR(0.21, tm.connections.dataForSynapse(syns4[0]).permanence,
              EPSILON);
}

TEST(TemporalMemoryTest, testSaveLoad) {
  TemporalMemory tm1(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  serializationTestPrepare(tm1);

  // Using binary streaming
  stringstream ss;
  tm1.save(ss);

  TemporalMemory tm2;
  tm2.load(ss);

  ASSERT_TRUE(tm1 == tm2);

  serializationTestVerify(tm2);
}

TEST(TemporalMemoryTest, testSaveArLoadAr) {
  TemporalMemory tm1(
      /*columnDimensions*/ {32},
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21f,
      /*connectedPermanence*/ 0.50f,
      /*minThreshold*/ 2,
      /*maxNewSynapseCount*/ 3,
      /*permanenceIncrement*/ 0.10f,
      /*permanenceDecrement*/ 0.10f,
      /*predictedSegmentDecrement*/ 0.0f,
      /*seed*/ 42);

  serializationTestPrepare(tm1);

  // Using Cereal Serialization
  stringstream ss1;
  tm1.save(ss1);

  TemporalMemory tm2;
  tm2.load(ss1);

  ASSERT_TRUE(tm1 == tm2);

  serializationTestVerify(tm2);

}


/*
 * Test compute( extraActive, extraWinners )
 
 * This test runs an artificial pattern through the TM.   At 10% column
 * sparsity, there are not enough active cells to reach the activation threshold
 * (12 < 13), unless the extra inputs are correctly included.  The extra inputs
 * are a copy of the current cell activity.
 */
TEST(TemporalMemoryTest, testExtraActive) {

  SDR columns({120});

  vector<SDR> pattern( 10, columns.dimensions );
  for(auto i = 0u; i < pattern.size(); i++) {
    Random rng( i + 99u );             // Use deterministic seeds for unit tests.
    auto &sdr = pattern[i];
    sdr.randomize( 0.10f, rng );
    auto &data = sdr.getSparse();
    std::sort(data.begin(), data.end());
  }

  auto tm = TemporalMemory(columns.dimensions,
    /* cellsPerColumn */               12,
    /* activationThreshold */          13,
    /* initialPermanence */            0.21f,
    /* connectedPermanence */          0.50f,
    /* minThreshold */                 10,
    /* maxNewSynapseCount */           20,
    /* permanenceIncrement */          0.10f,
    /* permanenceDecrement */          0.03f,
    /* predictedSegmentDecrement */    0.001f,
    /* seed */                         42,
    /* maxSegmentsPerCell */           255,
    /* maxSynapsesPerSegment */        255,
    /* checkInputs */                  true,
    /* extra */                        (UInt)(columns.size * 12u));
  auto tm_dimensions = tm.getColumnDimensions();
  tm_dimensions.push_back( static_cast<UInt>(tm.getCellsPerColumn()) );
  SDR extraActive( tm_dimensions );
  SDR extraWinners( tm_dimensions );

  // Look at the pattern.
  for(UInt trial = 0; trial < 20; trial++) {
    tm.reset();
    for(const auto &x : pattern) {
      // Calculate TM output
      tm.compute(x, true, extraActive, extraWinners);
      // update the external 'hints' for the next iteration
      tm.getActiveCells( extraActive );
      tm.getWinnerCells( extraWinners );
    }
    if( trial >= 19 ) {
      ASSERT_LT( tm.anomaly, 0.05f );
    }
  }

  // Test the test:  Verify that when the external inputs are missing this test
  // fails.
  tm.reset();
  extraActive.zero(); //zero out the external inputs
  extraWinners.zero();
  for(const auto &x : pattern) {
    // Predict whats going to happen.
    tm.activateDendrites(true, extraActive, extraWinners);
    auto predictedCells = tm.getPredictiveCells();
    ASSERT_TRUE( predictedCells.getSum() == 0 ); // No predictions, numActive < threshold
    // Calculate TM output
    tm.compute(x, true);
    ASSERT_GT( tm.anomaly, 0.95f );
  }
}

TEST(TemporalMemoryTest, testEquals) {
  TemporalMemory tm({10,10});
  auto tmCopy = tm;
  ASSERT_EQ(tm, tmCopy);

  SDR data({tm.getColumnDimensions()});
  data.setSparse(SDR_sparse_t{1,2,3,4,5,13,21,22,23,25,28,49,51,53,55,69});
  tm.compute(data, true);

  ASSERT_NE(tm, tmCopy);
  tmCopy.compute(data, true);
  ASSERT_EQ(tm, tmCopy);
}

TEST(TemporalMemoryTest, testIncorrectDefaultConstructor) {
  TemporalMemory tmFail; //default empty constructor is only used for deserialization
  SDR data1({0});
  EXPECT_ANY_THROW(tmFail.compute(data1, true));
  
  TemporalMemory tmOk({32} /*column dims must always be specified*/);
  SDR data2({tmOk.getColumnDimensions()});
  EXPECT_NO_THROW(tmOk.compute(data2, true));
}

// Uncomment these tests individually to save/load from a file.
// This is useful for ad-hoc testing of backwards-compatibility.

// TEST(TemporalMemoryTest, saveTestFile)
// {
//   TemporalMemory tm(
//     /*columnDimensions*/ {32},
//     /*cellsPerColumn*/ 4,
//     /*activationThreshold*/ 3,
//     /*initialPermanence*/ 0.21f,
//     /*connectedPermanence*/ 0.50f,
//     /*minThreshold*/ 2,
//     /*maxNewSynapseCount*/ 3,
//     /*permanenceIncrement*/ 0.10f,
//     /*permanenceDecrement*/ 0.10f,
//     /*predictedSegmentDecrement*/ 0.0f,
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
//     /*initialPermanence*/ 0.21f,
//     /*connectedPermanence*/ 0.50f,
//     /*minThreshold*/ 2,
//     /*maxNewSynapseCount*/ 3,
//     /*permanenceIncrement*/ 0.10f,
//     /*permanenceDecrement*/ 0.10f,
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
} // namespace
