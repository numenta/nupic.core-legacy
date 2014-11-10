/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of unit tests for Connections
 */

#include <iostream>
#include "ConnectionsTest.hpp"

using namespace std;
using namespace nta;
using namespace nta::algorithms::connections;

namespace nta {

  void ConnectionsTest::RunTests()
  {
    testCreateSegment();
    testCreateSynapse();
    testUpdateSynapsePermanence();
    testMostActiveSegmentForCells();
    testMostActiveSegmentForCellsNone();
    testComputeActivity();
    testActiveSegments();
  }

  /**
   * Creates a segment, and makes sure that it got created on the correct cell.
   */
  void ConnectionsTest::testCreateSegment()
  {
    Connections connections(1024);
    Segment segment;
    Cell cell(10);

    segment = connections.createSegment(cell);
    TESTEQUAL(segment.idx, 0);
    TESTEQUAL(segment.cell.idx, cell.idx);

    segment = connections.createSegment(cell);
    TESTEQUAL(segment.idx, 1);
    TESTEQUAL(segment.cell.idx, cell.idx);

    vector<Segment> segments = connections.segmentsForCell(cell);
    TESTEQUAL(segments.size(), 2);

    for (SegmentIdx i = 0; i < segments.size(); i++) {
      TESTEQUAL(segments[i].idx, i);
      TESTEQUAL(segments[i].cell.idx, cell.idx);
    }
  }

  /**
   * Creates a synapse, and makes sure that it got created on the correct
   * segment, and that its data was correctly stored.
   */
  void ConnectionsTest::testCreateSynapse()
  {
    Connections connections(1024);
    Cell cell(10), presynapticCell;
    Segment segment = connections.createSegment(cell);
    Synapse synapse;

    presynapticCell.idx = 50;
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);
    TESTEQUAL(synapse.idx, 0);
    TESTEQUAL(synapse.segment.idx, segment.idx);

    presynapticCell.idx = 150;
    synapse = connections.createSynapse(segment, presynapticCell, 0.48);
    TESTEQUAL(synapse.idx, 1);
    TESTEQUAL(synapse.segment.idx, segment.idx);

    vector<Synapse> synapses = connections.synapsesForSegment(segment);
    TESTEQUAL(synapses.size(), 2);

    for (SynapseIdx i = 0; i < synapses.size(); i++) {
      TESTEQUAL(synapses[i].idx, i);
      TESTEQUAL(synapses[i].segment.idx, segment.idx);
      TESTEQUAL(synapses[i].segment.cell.idx, cell.idx);
    }

    SynapseData synapseData;

    synapseData = connections.dataForSynapse(synapses[0]);
    TESTEQUAL(synapseData.presynapticCell.idx, 50);
    TESTEQUAL_FLOAT(synapseData.permanence, (Permanence)0.34);

    synapseData = connections.dataForSynapse(synapses[1]);
    TESTEQUAL(synapseData.presynapticCell.idx, 150);
    TESTEQUAL_FLOAT(synapseData.permanence, (Permanence)0.48);
  }

  /**
   * Creates a synapse and updates its permanence, and makes sure that its
   * data was correctly updated.
   */
  void ConnectionsTest::testUpdateSynapsePermanence()
  {
    Connections connections(1024);
    Cell cell(10), presynapticCell(50);
    Segment segment = connections.createSegment(cell);
    Synapse synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    connections.updateSynapsePermanence(synapse, 0.21);

    SynapseData synapseData = connections.dataForSynapse(synapse);
    TESTEQUAL_FLOAT(synapseData.permanence, (Real)0.21);
  }

  /**
   * Creates a sample set of connections, and makes sure that getting the most
   * active segment for a collection of cells returns the right segment.
   */
  void ConnectionsTest::testMostActiveSegmentForCells()
  {
    Connections connections(1024);
    Segment segment;
    Synapse synapse;
    Cell cell, presynapticCell;
    vector<Cell> cells;
    vector<Cell> input;

    cell.idx = 10; presynapticCell.idx = 150;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    cell.idx = 20; presynapticCell.idx = 50;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);

    Cell cell1(10), cell2(20);
    cells.push_back(cell1);
    cells.push_back(cell2);

    Cell input1(50);
    input.push_back(input1);

    bool result = connections.mostActiveSegmentForCells(
      cells, input, 0, segment);

    TESTEQUAL(result, true);

    TESTEQUAL(segment.cell.idx, 20);
    TESTEQUAL(segment.idx, 0);
  }

  /**
   * Creates a sample set of connections, and makes sure that getting the most
   * active segment for a collection of cells with no activity returns
   * no segment.
   */
  void ConnectionsTest::testMostActiveSegmentForCellsNone()
  {
    Connections connections(1024);
    Segment segment;
    Synapse synapse;
    Cell cell, presynapticCell;
    vector<Cell> cells;
    vector<Cell> input;

    cell.idx = 10; presynapticCell.idx = 150;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    cell.idx = 20; presynapticCell.idx = 50;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);

    Cell cell1(10), cell2(20);
    cells.push_back(cell1);
    cells.push_back(cell2);

    Cell input1(150);
    input.push_back(input1);

    bool result = connections.mostActiveSegmentForCells(
      cells, input, 2, segment);

    TESTEQUAL(result, false);
  }

  /**
   * Creates a sample set of connections, and makes sure that computing the
   * activity for a collection of cells with no activity returns the right
   * activity data.
   */
  void ConnectionsTest::testComputeActivity()
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    TESTEQUAL(activity.activeSegmentsForCell.size(), 1);
    cell.idx = 20;
    TESTEQUAL(activity.activeSegmentsForCell[cell].size(), 1);
    segment = activity.activeSegmentsForCell[cell][0];
    TESTEQUAL(segment.idx, 1);
    TESTEQUAL(segment.cell.idx, 20);

    TESTEQUAL(activity.numActiveSynapsesForSegment.size(), 3);
    segment.cell.idx = 10; segment.idx = 0;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 1);
    segment.cell.idx = 20; segment.idx = 0;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 1);
    segment.cell.idx = 20; segment.idx = 1;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 2);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * active segments from the computed activity.
   */
  void ConnectionsTest::testActiveSegments()
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    vector<Segment> activeSegments = connections.activeSegments(activity);

    TESTEQUAL(activeSegments.size(), 1);
    segment = activeSegments[0];
    TESTEQUAL(segment.idx, 1);
    TESTEQUAL(segment.cell.idx, 20);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * active cells from the computed activity.
   */
  void ConnectionsTest::testActiveCells()
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    vector<Cell> activeCells = connections.activeCells(activity);

    TESTEQUAL(activeCells.size(), 1);
    TESTEQUAL(activeCells[0].idx, 20);
  }

  void ConnectionsTest::setupSampleConnections(Connections &connections)
  {
    Segment segment;
    Synapse synapse;
    Cell cell, presynapticCell;

    cell.idx = 10;
    segment = connections.createSegment(cell);

    presynapticCell.idx = 150;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 151;
    synapse = connections.createSynapse(segment, presynapticCell, 0.15);

    cell.idx = 20;
    segment = connections.createSegment(cell);

    presynapticCell.idx = 50;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 51;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 52;
    synapse = connections.createSynapse(segment, presynapticCell, 0.15);

    segment = connections.createSegment(cell);

    presynapticCell.idx = 80;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 81;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 82;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    connections.updateSynapsePermanence(synapse, 0.15);
  }

  Activity ConnectionsTest::computeSampleActivity(Connections &connections)
  {
    Cell cell;
    vector<Cell> input;

    cell.idx = 150; input.push_back(cell);
    cell.idx = 151; input.push_back(cell);
    cell.idx = 50; input.push_back(cell);
    cell.idx = 52; input.push_back(cell);
    cell.idx = 80; input.push_back(cell);
    cell.idx = 81; input.push_back(cell);
    cell.idx = 82; input.push_back(cell);

    Activity activity = connections.computeActivity(input, 0.50, 2);
    return activity;
  }

} // end namespace nta
