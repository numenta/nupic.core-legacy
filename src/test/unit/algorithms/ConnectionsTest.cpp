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
    testGetMostActiveSegmentForCells();
    testGetMostActiveSegmentForCellsNone();
    testComputeActivity();
  }

  void ConnectionsTest::setup(Connections& connections)
  {
  }

  void ConnectionsTest::testCreateSegment()
  {
    Connections connections(1024);
    setup(connections);

    Segment segment;
    Cell cell = {10};

    segment = connections.createSegment(cell);
    TESTEQUAL(segment.idx, 0);
    TESTEQUAL(segment.cell.idx, cell.idx);

    segment = connections.createSegment(cell);
    TESTEQUAL(segment.idx, 1);
    TESTEQUAL(segment.cell.idx, cell.idx);

    vector<Segment> segments = connections.getSegmentsForCell(cell);
    TESTEQUAL(segments.size(), 2);

    for (SegmentIdx i = 0; i < segments.size(); i++) {
      TESTEQUAL(segments[i].idx, i);
      TESTEQUAL(segments[i].cell.idx, cell.idx);
    }
  }

  void ConnectionsTest::testCreateSynapse()
  {
    Connections connections(1024);
    setup(connections);

    Cell cell = {10}, presynapticCell;
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

    vector<Synapse> synapses = connections.getSynapsesForSegment(segment);
    TESTEQUAL(synapses.size(), 2);

    for (SynapseIdx i = 0; i < synapses.size(); i++) {
      TESTEQUAL(synapses[i].idx, i);
      TESTEQUAL(synapses[i].segment.idx, segment.idx);
      TESTEQUAL(synapses[i].segment.cell.idx, cell.idx);
    }

    SynapseData synapseData;

    synapseData = connections.getDataForSynapse(synapses[0]);
    TESTEQUAL(synapseData.presynapticCell.idx, 50);
    TESTEQUAL_FLOAT(synapseData.permanence, (Permanence)0.34);

    synapseData = connections.getDataForSynapse(synapses[1]);
    TESTEQUAL(synapseData.presynapticCell.idx, 150);
    TESTEQUAL_FLOAT(synapseData.permanence, (Permanence)0.48);
  }

  void ConnectionsTest::testUpdateSynapsePermanence()
  {
    Connections connections(1024);
    setup(connections);

    Cell cell = {10}, presynapticCell = {50};
    Segment segment = connections.createSegment(cell);
    Synapse synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    connections.updateSynapsePermanence(synapse, 0.21);

    SynapseData synapseData = connections.getDataForSynapse(synapse);
    TESTEQUAL_FLOAT(synapseData.permanence, (Real)0.21);
  }

  void ConnectionsTest::testGetMostActiveSegmentForCells()
  {
    Connections connections(1024);
    setup(connections);
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

    Cell cell1 = {10}, cell2 = {20};
    cells.push_back(cell1);
    cells.push_back(cell2);

    Cell input1 = {50};
    input.push_back(input1);

    bool result = connections.getMostActiveSegmentForCells(
      cells, input, 0, segment);

    TESTEQUAL(result, true);

    TESTEQUAL(segment.cell.idx, 20);
    TESTEQUAL(segment.idx, 0);
  }

  void ConnectionsTest::testGetMostActiveSegmentForCellsNone()
  {
    Connections connections(1024);
    setup(connections);
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

    Cell cell1 = {10}, cell2 = {20};
    cells.push_back(cell1);
    cells.push_back(cell2);

    Cell input1 = {150};
    input.push_back(input1);

    bool result = connections.getMostActiveSegmentForCells(
      cells, input, 2, segment);

    TESTEQUAL(result, false);
  }

  void ConnectionsTest::testComputeActivity()
  {
    Connections connections(1024);
    setup(connections);
    vector<Cell> input;
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
    synapse = connections.createSynapse(segment, presynapticCell, 0.15);

    cell.idx = 150; input.push_back(cell);
    cell.idx = 151; input.push_back(cell);
    cell.idx = 50; input.push_back(cell);
    cell.idx = 52; input.push_back(cell);
    cell.idx = 80; input.push_back(cell);
    cell.idx = 81; input.push_back(cell);
    cell.idx = 82; input.push_back(cell);

    Activity activity = connections.computeActivity(input, 0.50, 2);

    TESTEQUAL(activity.numActiveSegmentsForCell.size(), 1);
    cell.idx = 20;
    TESTEQUAL(activity.numActiveSegmentsForCell[cell], 1);

    TESTEQUAL(activity.numActiveSynapsesForSegment.size(), 3);
    segment.cell.idx = 10; segment.idx = 0;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 1);
    segment.cell.idx = 20; segment.idx = 0;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 1);
    segment.cell.idx = 20; segment.idx = 1;
    TESTEQUAL(activity.numActiveSynapsesForSegment[segment], 2);
  }

} // end namespace nta
