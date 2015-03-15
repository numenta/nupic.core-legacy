/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

  void TemporalMemoryTest::print_vec(vector<UInt> vec)
  {
    for (auto & elem : vec) {
      cout << elem << " ";
    }
    cout << endl;
  }

  void TemporalMemoryTest::print_vec(vector<Real> vec)
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

  bool TemporalMemoryTest::check_vector_eq(UInt arr[], vector<UInt> vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (arr[i] != vec[i]) {
        return false;
      }
    }
    return true;
  }

  bool TemporalMemoryTest::check_vector_eq(Real arr[], vector<Real> vec)
  {
    for (UInt i = 0; i < vec.size(); i++) {
      if (!almost_eq(arr[i],vec[i])) {
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

  bool TemporalMemoryTest::check_vector_eq(vector<UInt> vec1, vector<UInt> vec2)
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

  bool TemporalMemoryTest::check_set_eq(set<Segment>& vec1, set<Segment>& vec2)
  {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (Segment segment : vec2) {
      if (vec1.find(segment) == vec1.end()) {
        return false;
      }
    }
    return true;
  }

  void TemporalMemoryTest::check_spatial_eq(const TemporalMemory& tm1, const TemporalMemory& tm2)
  {
    NTA_CHECK(tm1.getNumColumns() == tm2.getNumColumns());
    NTA_CHECK(tm1.getCellsPerColumn() == tm2.getCellsPerColumn());
    NTA_CHECK(tm1.getActivationThreshold() == tm2.getActivationThreshold());
    NTA_CHECK(tm1.getLearningRadius() == tm2.getLearningRadius());
    NTA_CHECK(tm1.getMinThreshold() == tm2.getMinThreshold());
    NTA_CHECK(tm1.getMaxNewSynapseCount() == tm2.getMaxNewSynapseCount());
    TEST(nupic::nearlyEqual(tm1.getInitialPermanence(), tm2.getInitialPermanence()));
    TEST(nupic::nearlyEqual(tm1.getConnectedPermanence(), tm2.getConnectedPermanence()));
    TEST(nupic::nearlyEqual(tm1.getPermanenceIncrement(), tm2.getPermanenceIncrement()));
    TEST(nupic::nearlyEqual(tm1.getPermanenceDecrement(), tm2.getPermanenceDecrement()));
  }

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
    testBurstColumns();
    testBurstColumnsEmpty();
    //testLearnOnSegments();
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
    //testSaveLoad();

  }

  void TemporalMemoryTest::testInitInvalidParams()
  {
    TemporalMemory tm;

    // Invalid columnDimensions
    vector<UInt> columnDim;
    SHOULDFAIL(tm.initialize(columnDim, 32));

    // Invalid cellsPerColumn
    columnDim.push_back(2048);
    SHOULDFAIL(tm.initialize(columnDim, 0));
    SHOULDFAIL(tm.initialize(columnDim, -10));
  }

  void TemporalMemoryTest::testActivateCorrectlyPredictiveCells()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    set<Cell> prevPredictiveCells = { Cell(0), Cell(237), Cell(1026), Cell(26337), Cell(26339), Cell(55536) };
    set<UInt> activeColumns = { 32, 47, 823 };

    tm.activateCorrectlyPredictiveCells(prevPredictiveCells, activeColumns);

    set<Cell> expectedCells = { Cell(1026), Cell(26337), Cell(26339) };
    set<UInt> expectedCols = { 32, 823 };
    NTA_CHECK(check_set_eq(tm.activeCells_, expectedCells));
    NTA_CHECK(check_set_eq(tm.winnerCells_, expectedCells));
    NTA_CHECK(check_set_eq(tm.predictedColumns_, expectedCols));
  }

  void TemporalMemoryTest::testActivateCorrectlyPredictiveCellsEmpty()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    {
      set<Cell> prevPredictiveCells = {};
      set<UInt> activeColumns = {};

      tm.activateCorrectlyPredictiveCells(prevPredictiveCells, activeColumns);

      set<Cell> expectedCells = {};
      set<UInt> expectedCols = {};
      NTA_CHECK(check_set_eq(tm.activeCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.winnerCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.predictedColumns_, expectedCols));
    }

    // No previous predictive cells

    {
      set<Cell> prevPredictiveCells = {};
      set<UInt> activeColumns = { 32, 47, 823 };

      tm.activateCorrectlyPredictiveCells(prevPredictiveCells, activeColumns);

      set<Cell> expectedCells = {};
      set<UInt> expectedCols = {};
      NTA_CHECK(check_set_eq(tm.activeCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.winnerCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.predictedColumns_, expectedCols));
    }

    // No active columns

    {
      set<Cell> prevPredictiveCells = { Cell(0), Cell(237), Cell(1026), Cell(26337), Cell(26339), Cell(55536) };
      set<UInt> activeColumns = { };

      tm.activateCorrectlyPredictiveCells(prevPredictiveCells, activeColumns);

      set<Cell> expectedCells = {};
      set<UInt> expectedCols = {};
      NTA_CHECK(check_set_eq(tm.activeCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.winnerCells_, expectedCells));
      NTA_CHECK(check_set_eq(tm.predictedColumns_, expectedCols));
    }
  }

  void TemporalMemoryTest::testBurstColumns()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    tm.setCellsPerColumn(4);
    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);
    tm.reSeed(42);
    //tm.reset();

    Connections connections = *tm.connections_;
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

    tm.burstColumns(activeColumns, predictiveCols, prevActiveCells, prevWinnerCells, connections);

    set<Cell> expectedActiveCells = { Cell(0), Cell(1), Cell(2), Cell(3), Cell(4), Cell(5), Cell(6), Cell(7) };
    set<Cell> expectedWinnerCells = { Cell(0), Cell(6) }; // 6 is randomly chosen cell
    set<Segment> expectedLearningSegments = { Segment(0, Cell(0)), Segment(0, Cell(6)) };
    NTA_CHECK(check_set_eq(tm.activeCells_, expectedActiveCells));
    NTA_CHECK(check_set_eq(tm.winnerCells_, expectedWinnerCells));
    NTA_CHECK(check_set_eq(tm.learningSegments_, expectedLearningSegments));

    // Check that new segment was added to winner cell(6) in column 1
    vector<Segment> segments = connections.segmentsForCell(6);
    vector<Segment> expectedSegments = { Segment(0, Cell(6)) };
    NTA_CHECK(check_vector_eq(segments, expectedSegments));
  }

  void TemporalMemoryTest::testBurstColumnsEmpty()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    set<UInt> activeColumns = { };
    set<UInt> predictiveCols = { };
    set<Cell> prevActiveCells = { };
    set<Cell> prevWinnerCells = { };
    Connections connections = *tm.connections_;

    tm.burstColumns(activeColumns, predictiveCols, prevActiveCells, prevWinnerCells, connections);

    set<Cell> expectedActiveCells = { };
    set<Cell> expectedWinnerCells = { };
    set<Segment> expectedLearningSegments = { };
    NTA_CHECK(check_set_eq(tm.activeCells_, expectedActiveCells));
    NTA_CHECK(check_set_eq(tm.winnerCells_, expectedWinnerCells));
    NTA_CHECK(check_set_eq(tm.learningSegments_, expectedLearningSegments));
  }

  void TemporalMemoryTest::testLearnOnSegments()
  {
    vector<Synapse> synapses;
    bool eq;
    TemporalMemory tm;
    setup(tm, 2048);

    tm.setMaxNewSynapseCount(2);
    tm.reset();

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    segment = connections.createSegment(Cell(1));
    connections.createSynapse(segment, Cell(733), 0.7);

    segment = connections.createSegment(Cell(8));
    connections.createSynapse(segment, Cell(486), 0.9);

    segment = connections.createSegment(Cell(100));

    set<Segment> prevActiveSegments = { Segment(0,Cell(0)), Segment(2,Cell(8)) };
    set<Segment> learningSegments = { Segment(1,Cell(1)), Segment(3,Cell(100)) };
    set<Cell> prevActiveCells = { Cell(23), Cell(37), Cell(733) };
    set<Cell> winnerCells = { Cell(0) };
    set<Cell> prevWinnerCells = { Cell(10), Cell(11), Cell(12), Cell(13), Cell(14) };

    tm.learnOnSegments(
      prevActiveSegments,
      learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      connections);

    // Check segment 0
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, Segment(0, Cell(0)))).permanence, Real(0.7));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(1, Segment(0, Cell(0)))).permanence, Real(0.5));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(2, Segment(0, Cell(0)))).permanence, Real(0.8));
    EXPECT_TRUE(eq);

    // Check segment 1
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(3, Segment(1, Cell(1)))).permanence, Real(0.8));
    EXPECT_TRUE(eq);
    synapses = connections.synapsesForSegment(Segment(1, Cell(1)));
    ASSERT_EQ(synapses.size(), 2);

    // Check segment 2
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(4, Segment(2, Cell(8)))).permanence, Real(0.9));
    EXPECT_TRUE(eq);
    synapses = connections.synapsesForSegment(Segment(2, Cell(8)));
    ASSERT_EQ(synapses.size(), 1);

    // Check segment 3
    synapses = connections.synapsesForSegment(Segment(3, Cell(100)));
    ASSERT_EQ(synapses.size(), 2);
  }

  void TemporalMemoryTest::testComputePredictiveCells()
  {
    TemporalMemory tm;
    setup(tm, 2048);

    tm.setMaxNewSynapseCount(2);
    tm.reset();

    Connections connections = *tm.connections_;
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

    tm.computePredictiveCells(activeCells, connections);

    set<Segment> expectedActiveSegments = { };
    set<Cell> expectedPredictiveCells = { };
    NTA_CHECK(check_set_eq(tm.activeSegments_, expectedActiveSegments));
    NTA_CHECK(check_set_eq(tm.predictiveCells_, expectedPredictiveCells));
  }

  void TemporalMemoryTest::testBestMatchingCell()
  {
    Cell bestCell;
    Segment bestSegment;
    TemporalMemory tm;
    setup(tm, 2048);

    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);
    tm.reSeed(42);

    Connections connections = *tm.connections_;

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

    tie(bestCell, bestSegment) = tm.bestMatchingCell(tm.cellsForColumn(0), activeCells, connections);
    ASSERT_EQ(bestCell, Cell(0));
    ASSERT_EQ(bestSegment, Segment(0, Cell(0)));

    tie(bestCell, bestSegment) = tm.bestMatchingCell(tm.cellsForColumn(3), activeCells, connections);
    ASSERT_EQ(bestCell, Cell(96)); // Random cell from column

    tie(bestCell, bestSegment) = tm.bestMatchingCell(tm.cellsForColumn(999), activeCells, connections);
    ASSERT_EQ(bestCell, Cell(31972)); // Random cell from column
  }

  void TemporalMemoryTest::testBestMatchingCellFewestSegments()
  {
    Cell cell(0);
    Segment segment;
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2}, 2);
    tm.setMinThreshold(1);

    Connections connections = *tm.connections_;
    segment = connections.createSegment(cell);
    connections.createSynapse(segment, 3, 0.3);

    set<Cell> activeSynapsesForSegment = { };

    for (int i = 0; i < 100; i++)
    {
      // Never pick cell 0, always pick cell 1
      tie(cell, segment) = tm.bestMatchingCell(tm.cellsForColumn(0),
        activeSynapsesForSegment,
        connections);
      ASSERT_EQ(cell, Cell(1));
    }
  }

  void TemporalMemoryTest::testBestMatchingSegment()
  {
    Int numActiveSynapses;
    Segment bestSegment;
    TemporalMemory tm;
    setup(tm, 2048);

    tm.setConnectedPermanence(0.50);
    tm.setMinThreshold(1);

    Connections connections = *tm.connections_;

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

    tie(bestSegment, numActiveSynapses) = tm.bestMatchingSegment(Cell(0), activeCells, connections);
    ASSERT_EQ(bestSegment, Segment(0, Cell(0)));
    ASSERT_EQ(numActiveSynapses, 2);

    tie(bestSegment, numActiveSynapses) = tm.bestMatchingSegment(Cell(1), activeCells, connections);
    ASSERT_EQ(bestSegment, Segment(0, Cell(1)));
    ASSERT_EQ(numActiveSynapses, 1);

    tie(bestSegment, numActiveSynapses) = tm.bestMatchingSegment(Cell(8), activeCells, connections);
    ASSERT_EQ(bestSegment, Segment(-1, Cell(0)));
    ASSERT_EQ(numActiveSynapses, 0);

    tie(bestSegment, numActiveSynapses) = tm.bestMatchingSegment(Cell(100), activeCells, connections);
    ASSERT_EQ(bestSegment, Segment(-1, Cell(0)));
    ASSERT_EQ(numActiveSynapses, 0);
  }

  void TemporalMemoryTest::testLeastUsedCell()
  {
    Cell cell(0);

    TemporalMemory tm;
    tm.initialize(vector<UInt>{2}, 2);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(cell);
    connections.createSynapse(segment, 3, 0.3);

    set<Cell> cells = {};

    for (int i = 0; i < 100; i++)
    {
      // Never pick cell 0, always pick cell 1
      tie(cell, segment) = tm.bestMatchingCell(tm.cellsForColumn(0), cells, connections);
      ASSERT_EQ(cell, Cell(1));
    }
  }

  void TemporalMemoryTest::testAdaptSegment()
  {
    TemporalMemory tm;
    vector<Synapse> synapses;
    bool eq;

    setup(tm, 2048);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.6);
    connections.createSynapse(segment, Cell(37), 0.4);
    connections.createSynapse(segment, Cell(477), 0.9);

    synapses = vector<Synapse>{ Synapse(0, segment), Synapse(1, segment) };
    tm.adaptSegment(segment, synapses, connections);
    
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Real(0.7));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(1, segment)).permanence, Real(0.5));
    EXPECT_TRUE(eq);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(2, segment)).permanence, Real(0.8));
    EXPECT_TRUE(eq);
  }

  void TemporalMemoryTest::testAdaptSegmentToMax()
  {
    TemporalMemory tm;
    vector<Synapse> synapses;
    bool eq;

    setup(tm, 2048);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));
    synapses.push_back(connections.createSynapse(segment, Cell(23), 0.9));

    tm.adaptSegment(segment, synapses, connections);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Real(1.0));
    EXPECT_TRUE(eq);

    // Now permanence should be at min
    tm.adaptSegment(segment, synapses, connections);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Real(1.0));
    EXPECT_TRUE(eq);
  }

  void TemporalMemoryTest::testAdaptSegmentToMin()
  {
    TemporalMemory tm;
    vector<Synapse> synapses = { Synapse(-1, Segment(-1, Cell(0))) };
    bool eq;

    setup(tm, 2048);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, Cell(23), 0.1);

    tm.adaptSegment(segment, synapses, connections);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Real(0.0));
    EXPECT_TRUE(eq);

    // Now permanence should be at min
    tm.adaptSegment(segment, synapses, connections);
    eq = nupic::nearlyEqual(connections.dataForSynapse(Synapse(0, segment)).permanence, Real(0.0));
    EXPECT_TRUE(eq);
  }

  void TemporalMemoryTest::testPickCellsToLearnOn()
  {
    TemporalMemory tm;

    setup(tm, 2048);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));

    set<Cell> winnerCells = { Cell(4), Cell(47), Cell(58), Cell(93) };
    set<Cell> expectedCells;

    expectedCells = set<Cell> { Cell(4), Cell(47) }; // Randomly picked
    NTA_CHECK(check_set_eq(tm.pickCellsToLearnOn(2, segment, winnerCells, connections), expectedCells));

    expectedCells = set<Cell> { Cell(4), Cell(47), Cell(58), Cell(93) };
    NTA_CHECK(check_set_eq(tm.pickCellsToLearnOn(100, segment, winnerCells, connections), expectedCells));

    expectedCells = set<Cell> { };
    NTA_CHECK(check_set_eq(tm.pickCellsToLearnOn(0, segment, winnerCells, connections), expectedCells));
  }

  void TemporalMemoryTest::testPickCellsToLearnOnAvoidDuplicates()
  {
    TemporalMemory tm;

    setup(tm, 2048);

    Connections connections = *tm.connections_;
    Segment segment = connections.createSegment(Cell(0));
    connections.createSynapse(segment, 23, 0.6);

    set<Cell> winnerCells = { Cell(23) };

    // Ensure that no additional(duplicate) cells were picked
    set<Cell> expectedCells = {};
    NTA_CHECK(check_set_eq(tm.pickCellsToLearnOn(2, segment, winnerCells, connections), expectedCells));
  }

  void TemporalMemoryTest::testColumnForCell1D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    TEST(tm.columnForCell(Cell(0)) == 0);
    TEST(tm.columnForCell(Cell(4)) == 0);
    TEST(tm.columnForCell(Cell(5)) == 1);
    TEST(tm.columnForCell(Cell(10239)) == 2047);
  }

  void TemporalMemoryTest::testColumnForCell2D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    TEST(tm.columnForCell(Cell(0)) == 0);
    TEST(tm.columnForCell(Cell(3)) == 0);
    TEST(tm.columnForCell(Cell(4)) == 1);
    TEST(tm.columnForCell(Cell(16383)) == 4095);
  }

  void TemporalMemoryTest::testColumnForCellInvalidCell()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    EXPECT_NO_THROW(tm.columnForCell(Cell(16383)));
    EXPECT_THROW(tm.columnForCell(Cell(16384)), std::exception);
    EXPECT_THROW(tm.columnForCell(Cell(-1)), std::exception);
  }

  void TemporalMemoryTest::testCellsForColumn1D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{2048}, 5);

    vector<Cell> expectedCells = { Cell(5), Cell(6), Cell(7), Cell(8), Cell(9) };
    NTA_CHECK(check_vector_eq(tm.cellsForColumn(1), expectedCells));
  }
  
  void TemporalMemoryTest::testCellsForColumn2D()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64, 64}, 4);

    vector<Cell> expectedCells = { Cell(256), Cell(257), Cell(258), Cell(259) };
    NTA_CHECK(check_vector_eq(tm.cellsForColumn(64), expectedCells));
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

    ASSERT_EQ(tm.numberOfColumns(), 64 * 64);
  }

  void TemporalMemoryTest::testNumberOfCells()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{64,64}, 32);

    ASSERT_EQ(tm.numberOfCells(), 64 * 64 * 32);
  }

  void TemporalMemoryTest::testMapCellsToColumns()
  {
    TemporalMemory tm;
    tm.initialize(vector<UInt>{100}, 4);

    set<Cell> cells = { Cell(0), Cell(1), Cell(2), Cell(5), Cell(399) };
    map<Int, vector<Cell>> columnsForCells = tm.mapCellsToColumns(cells);

    vector<Cell> expectedCells = { Cell(0), Cell(1), Cell(2) };
    ASSERT_EQ(columnsForCells[0], expectedCells);
    ASSERT_EQ(columnsForCells[1], vector<Cell>{Cell(5)});
    ASSERT_EQ(columnsForCells[99], vector<Cell>{Cell(399)});
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

} // end namespace nupic
