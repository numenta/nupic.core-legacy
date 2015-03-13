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
    NTA_CHECK(check_vector_eq(connections.segmentsForCell(6), vector<Segment>{ Segment(0, Cell(6)) }));
  }

  void TemporalMemoryTest::testBurstColumnsEmpty()
  {
    /*
    tm = self.tm

      activeColumns = set()
      predictedColumns = set()
      prevActiveCells = set()
      prevWinnerCells = set()
      connections = tm.connections

      (activeCells,
      winnerCells,
      learningSegments) = tm.burstColumns(activeColumns,
      predictedColumns,
      prevActiveCells,
      prevWinnerCells,
      connections)

      self.assertEqual(activeCells, set())
      self.assertEqual(winnerCells, set())
      self.assertEqual(learningSegments, set())
      */
  }

  void TemporalMemoryTest::testLearnOnSegments()
  {
    /*
    tm = TemporalMemory(maxNewSynapseCount = 2)

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)
      connections.createSynapse(0, 37, 0.4)
      connections.createSynapse(0, 477, 0.9)

      connections.createSegment(1)
      connections.createSynapse(1, 733, 0.7)

      connections.createSegment(8)
      connections.createSynapse(2, 486, 0.9)

      connections.createSegment(100)

      prevActiveSegments = set([0, 2])
      learningSegments = set([1, 3])
      prevActiveCells = set([23, 37, 733])
      winnerCells = set([0])
      prevWinnerCells = set([10, 11, 12, 13, 14])

      tm.learnOnSegments(prevActiveSegments,
      learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      connections)

      # Check segment 0
      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 0.7)

      (_, _, permanence) = connections.dataForSynapse(1)
      self.assertAlmostEqual(permanence, 0.5)

      (_, _, permanence) = connections.dataForSynapse(2)
      self.assertAlmostEqual(permanence, 0.8)

      # Check segment 1
      (_, _, permanence) = connections.dataForSynapse(3)
      self.assertAlmostEqual(permanence, 0.8)

      self.assertEqual(len(connections.synapsesForSegment(1)), 2)

      # Check segment 2
      (_, _, permanence) = connections.dataForSynapse(4)
      self.assertAlmostEqual(permanence, 0.9)

      self.assertEqual(len(connections.synapsesForSegment(2)), 1)

      # Check segment 3
      self.assertEqual(len(connections.synapsesForSegment(3)), 2)
      */
  }

  void TemporalMemoryTest::testComputePredictiveCells()
  {
    /*
    tm = TemporalMemory(activationThreshold = 2)

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)
      connections.createSynapse(0, 37, 0.5)
      connections.createSynapse(0, 477, 0.9)

      connections.createSegment(1)
      connections.createSynapse(1, 733, 0.7)
      connections.createSynapse(1, 733, 0.4)

      connections.createSegment(1)
      connections.createSynapse(2, 974, 0.9)

      connections.createSegment(8)
      connections.createSynapse(3, 486, 0.9)

      connections.createSegment(100)

      activeCells = set([23, 37, 733, 974])

      (activeSegments,
      predictiveCells) = tm.computePredictiveCells(activeCells, connections)
      self.assertEqual(activeSegments, set([0]))
      self.assertEqual(predictiveCells, set([0]))
      */
  }

  void TemporalMemoryTest::testBestMatchingCell()
  {
    /*
    tm = TemporalMemory(
      connectedPermanence = 0.50,
      minThreshold = 1,
      seed = 42
      )

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)
      connections.createSynapse(0, 37, 0.4)
      connections.createSynapse(0, 477, 0.9)

      connections.createSegment(0)
      connections.createSynapse(1, 49, 0.9)
      connections.createSynapse(1, 3, 0.8)

      connections.createSegment(1)
      connections.createSynapse(2, 733, 0.7)

      connections.createSegment(108)
      connections.createSynapse(3, 486, 0.9)

      activeCells = set([23, 37, 49, 733])

      self.assertEqual(tm.bestMatchingCell(tm.cellsForColumn(0),
      activeCells,
      connections),
      (0, 0))

      self.assertEqual(tm.bestMatchingCell(tm.cellsForColumn(3), # column containing cell 108
      activeCells,
      connections),
      (96, None))  # Random cell from column

      self.assertEqual(tm.bestMatchingCell(tm.cellsForColumn(999),
      activeCells,
      connections),
      (31972, None))  # Random cell from column
      */
  }

  void TemporalMemoryTest::testBestMatchingCellFewestSegments()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [2],
      cellsPerColumn = 2,
      connectedPermanence = 0.50,
      minThreshold = 1,
      seed = 42
      )

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 3, 0.3)

      activeSynapsesForSegment = set([])

      for _ in range(100) :
        # Never pick cell 0, always pick cell 1
        (cell, _) = tm.bestMatchingCell(tm.cellsForColumn(0),
        activeSynapsesForSegment,
        connections)
        self.assertEqual(cell, 1)
        */
  }

  void TemporalMemoryTest::testBestMatchingSegment()
  {
    /*
    tm = TemporalMemory(
      connectedPermanence = 0.50,
      minThreshold = 1
      )

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)
      connections.createSynapse(0, 37, 0.4)
      connections.createSynapse(0, 477, 0.9)

      connections.createSegment(0)
      connections.createSynapse(1, 49, 0.9)
      connections.createSynapse(1, 3, 0.8)

      connections.createSegment(1)
      connections.createSynapse(2, 733, 0.7)

      connections.createSegment(8)
      connections.createSynapse(3, 486, 0.9)

      activeCells = set([23, 37, 49, 733])

      self.assertEqual(tm.bestMatchingSegment(0,
      activeCells,
      connections),
      (0, 2))

      self.assertEqual(tm.bestMatchingSegment(1,
      activeCells,
      connections),
      (2, 1))

      self.assertEqual(tm.bestMatchingSegment(8,
      activeCells,
      connections),
      (None, None))

      self.assertEqual(tm.bestMatchingSegment(100,
      activeCells,
      connections),
      (None, None))
      */
  }

  void TemporalMemoryTest::testLeastUsedCell()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [2],
      cellsPerColumn = 2,
      seed = 42
      )

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 3, 0.3)

      for _ in range(100) :
        # Never pick cell 0, always pick cell 1
        self.assertEqual(tm.leastUsedCell(tm.cellsForColumn(0),
        connections),
        1)
        */
  }

  void TemporalMemoryTest::testAdaptSegment()
  {
    /*
    tm = self.tm

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)
      connections.createSynapse(0, 37, 0.4)
      connections.createSynapse(0, 477, 0.9)

      tm.adaptSegment(0, set([0, 1]), connections)

      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 0.7)

      (_, _, permanence) = connections.dataForSynapse(1)
      self.assertAlmostEqual(permanence, 0.5)

      (_, _, permanence) = connections.dataForSynapse(2)
      self.assertAlmostEqual(permanence, 0.8)
      */
  }

  void TemporalMemoryTest::testAdaptSegmentToMax()
  {
    /*
    tm = self.tm

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.9)

      tm.adaptSegment(0, set([0]), connections)
      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 1.0)

      # Now permanence should be at max
      tm.adaptSegment(0, set([0]), connections)
      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 1.0)
      */
  }

  void TemporalMemoryTest::testAdaptSegmentToMin()
  {
    /*
    tm = self.tm

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.1)

      tm.adaptSegment(0, set(), connections)
      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 0.0)

      # Now permanence should be at min
      tm.adaptSegment(0, set(), connections)
      (_, _, permanence) = connections.dataForSynapse(0)
      self.assertAlmostEqual(permanence, 0.0)
      */
  }

  void TemporalMemoryTest::testPickCellsToLearnOn()
  {
    /*
    tm = TemporalMemory(seed = 42)

      connections = tm.connections
      connections.createSegment(0)

      winnerCells = set([4, 47, 58, 93])

      self.assertEqual(tm.pickCellsToLearnOn(2, 0, winnerCells, connections),
      set([4, 58]))  # randomly picked

      self.assertEqual(tm.pickCellsToLearnOn(100, 0, winnerCells, connections),
      set([4, 47, 58, 93]))

      self.assertEqual(tm.pickCellsToLearnOn(0, 0, winnerCells, connections),
      set())
      */
  }

  void TemporalMemoryTest::testPickCellsToLearnOnAvoidDuplicates()
  {
    /*
    tm = TemporalMemory(seed = 42)

      connections = tm.connections
      connections.createSegment(0)
      connections.createSynapse(0, 23, 0.6)

      winnerCells = set([23])

      # Ensure that no additional(duplicate) cells were picked
      self.assertEqual(tm.pickCellsToLearnOn(2, 0, winnerCells, connections),
      set())
      */
  }

  void TemporalMemoryTest::testColumnForCell1D()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [2048],
      cellsPerColumn = 5
      )
      self.assertEqual(tm.columnForCell(0), 0)
      self.assertEqual(tm.columnForCell(4), 0)
      self.assertEqual(tm.columnForCell(5), 1)
      self.assertEqual(tm.columnForCell(10239), 2047)
      */
  }

  void TemporalMemoryTest::testColumnForCell2D()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 4
      )
      self.assertEqual(tm.columnForCell(0), 0)
      self.assertEqual(tm.columnForCell(3), 0)
      self.assertEqual(tm.columnForCell(4), 1)
      self.assertEqual(tm.columnForCell(16383), 4095)
      */
  }

  void TemporalMemoryTest::testColumnForCellInvalidCell()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 4
      )

      try :
      tm.columnForCell(16383)
      except IndexError :
    self.fail("IndexError raised unexpectedly")

      args = [16384]
      self.assertRaises(IndexError, tm.columnForCell, *args)

      args = [-1]
      self.assertRaises(IndexError, tm.columnForCell, *args)
      */
  }

  void TemporalMemoryTest::testCellsForColumn1D()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [2048],
      cellsPerColumn = 5
      )
      expectedCells = set([5, 6, 7, 8, 9])
      self.assertEqual(tm.cellsForColumn(1), expectedCells)
      */
  }
  
  void TemporalMemoryTest::testCellsForColumn2D()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 4
      )
      expectedCells = set([256, 257, 258, 259])
      self.assertEqual(tm.cellsForColumn(64), expectedCells)
      */
  }

  void TemporalMemoryTest::testCellsForColumnInvalidColumn()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 4
      )

      try :
      tm.cellsForColumn(4095)
      except IndexError :
    self.fail("IndexError raised unexpectedly")

      args = [4096]
      self.assertRaises(IndexError, tm.cellsForColumn, *args)

      args = [-1]
      self.assertRaises(IndexError, tm.cellsForColumn, *args)
      */
  }

  void TemporalMemoryTest::testNumberOfColumns()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 32
      )
      self.assertEqual(tm.numberOfColumns(), 64 * 64)
      */
  }

  void TemporalMemoryTest::testNumberOfCells()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [64, 64],
      cellsPerColumn = 32
      )
      self.assertEqual(tm.numberOfCells(), 64 * 64 * 32)
      */
  }

  void TemporalMemoryTest::testMapCellsToColumns()
  {
    /*
    tm = TemporalMemory(
      columnDimensions = [100],
      cellsPerColumn = 4
      )
      columnsForCells = tm.mapCellsToColumns(set([0, 1, 2, 5, 399]))
      self.assertEqual(columnsForCells[0], set([0, 1, 2]))
      self.assertEqual(columnsForCells[1], set([5]))
      self.assertEqual(columnsForCells[99], set([399]))
      */
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
    outfile.open(filename);
    tm1.save(outfile);
    outfile.close();

    ifstream infile(filename);
    tm2.load(infile);
    infile.close();

    check_spatial_eq(tm1, tm2);

    int ret = ::remove(filename);
    NTA_CHECK(ret == 0) << "Failed to delete " << filename;
  }

} // end namespace nupic
