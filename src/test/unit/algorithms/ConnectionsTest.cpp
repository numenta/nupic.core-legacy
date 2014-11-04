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
    Connections connections;
    setup(connections);

    Segment segment;
    connections.createSegment(10, segment);

    NTA_ASSERT(segment.cell == 10);
    NTA_ASSERT(segment.synapses.size() == 0);
  }

  void ConnectionsTest::testCreateSynapse()
  {
    Connections connections;
    setup(connections);

    Segment segment;
    connections.createSegment(10, segment);

    Synapse synapse;
    connections.createSynapse(segment, 50, 0.34, synapse);

    NTA_ASSERT(synapse.segment == &segment);
    NTA_ASSERT(synapse.presynapticCell == 50);
    NTA_ASSERT(nearlyEqual(synapse.permanence, (Real)0.34));

    NTA_ASSERT(segment.synapses.size() == 1);
    NTA_ASSERT(segment.synapses.front() == &synapse);
  }

  void ConnectionsTest::testUpdateSynapsePermanence()
  {
    Connections connections;
    setup(connections);

    Segment segment;
    connections.createSegment(10, segment);

    Synapse synapse;
    connections.createSynapse(segment, 50, 0.34, synapse);

    connections.updateSynapsePermanence(synapse, 0.21);
    NTA_ASSERT(nearlyEqual(synapse.permanence, (Real)0.21));
  }

  void ConnectionsTest::testGetMostActiveSegmentForCells()
  {
    Connections connections;
    setup(connections);
    Segment segment;
    Synapse synapse;
    vector<UInt> cells;
    vector<UInt> input;

    connections.createSegment(10, segment);
    connections.createSynapse(segment, 50, 0.34, synapse);

    connections.createSegment(20, segment);
    connections.createSynapse(segment, 150, 0.85, synapse);

    // cells.push_back(10);
    // cells.push_back(20);

    // input.push_back(50);

    bool result = connections.getMostActiveSegmentForCells(
      cells, input, 0, segment);

    NTA_ASSERT(result == false);
  }

  void ConnectionsTest::testComputeActivity()
  {
    Connections connections;
    setup(connections);
    vector<UInt> input;
    input.push_back(10);
    input.push_back(20);

    CellActivity activity;
    connections.computeActivity(input, 0.10, 5, activity);
    // TODO: Add assertion
  }

} // end namespace nta
