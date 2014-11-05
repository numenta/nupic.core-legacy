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
#include <nta/math/Math.hpp>
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
    NTA_ASSERT(segment.idx == 0);
    NTA_ASSERT(segment.cell.idx == cell.idx);

    segment = connections.createSegment(cell);
    NTA_ASSERT(segment.idx == 1);
    NTA_ASSERT(segment.cell.idx == cell.idx);

    vector<Segment> segments = connections.getSegmentsForCell(cell);
    NTA_ASSERT(segments.size() == 2);

    for (SegmentIdx i = 0; i < segments.size(); i++) {
      NTA_ASSERT(segments[i].idx == i);
      NTA_ASSERT(segments[i].cell.idx == cell.idx);
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
    NTA_ASSERT(synapse.idx == 0);
    NTA_ASSERT(synapse.segment.idx == segment.idx);

    presynapticCell.idx = 150;
    synapse = connections.createSynapse(segment, presynapticCell, 0.48);
    NTA_ASSERT(synapse.idx == 1);
    NTA_ASSERT(synapse.segment.idx == segment.idx);

    vector<Synapse> synapses = connections.getSynapsesForSegment(segment);
    NTA_ASSERT(synapses.size() == 2);

    for (SynapseIdx i = 0; i < synapses.size(); i++) {
      NTA_ASSERT(synapses[i].idx == i);
      NTA_ASSERT(synapses[i].segment.idx == segment.idx);
      NTA_ASSERT(synapses[i].segment.cell.idx == cell.idx);
    }

    SynapseData synapseData;

    synapseData = connections.getDataForSynapse(synapses[0]);
    NTA_ASSERT(synapseData.presynapticCell.idx == 50);
    NTA_ASSERT(nearlyEqual(synapseData.permanence, (Permanence)0.34));

    synapseData = connections.getDataForSynapse(synapses[1]);
    NTA_ASSERT(synapseData.presynapticCell.idx == 150);
    NTA_ASSERT(nearlyEqual(synapseData.permanence, (Permanence)0.48));
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
    NTA_ASSERT(nearlyEqual(synapseData.permanence, (Real)0.21));
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

    cell.idx = 10; presynapticCell.idx = 50;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    cell.idx = 20; presynapticCell.idx = 150;
    segment = connections.createSegment(cell);
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);

    // TODO: Enable test
    // cells.push_back(10);
    // cells.push_back(20);

    // input.push_back(50);

    bool result = connections.getMostActiveSegmentForCells(
      cells, input, 0, segment);

    NTA_ASSERT(result == false);
  }

  void ConnectionsTest::testComputeActivity()
  {
    Connections connections(1024);
    setup(connections);
    vector<Cell> input;
    Cell cell;

    for (CellIdx i = 10; i <= 20; i += 10) {
      cell.idx = i;
      input.push_back(cell);
    }

    Activity activity = connections.computeActivity(input, 0.10, 5);
    // TODO: Add assertion
  }

} // end namespace nta
