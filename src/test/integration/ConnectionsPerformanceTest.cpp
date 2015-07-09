/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of performance tests for Connections
 */

#include <fstream>
#include <iostream>
#include <time.h>
#include "ConnectionsPerformanceTest.hpp"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;

namespace nupic {

  void ConnectionsPerformanceTest::RunTests()
  {
    testTemporalMemoryUsage();
  }

  /**
   * Tests typical usage of Connections with Temporal Memory.
   */
  void ConnectionsPerformanceTest::testTemporalMemoryUsage()
  {
    // TODO: Implement actual test, this is just a placeholder for now
    clock_t start = clock();
    Connections connections(2048);
    setupSampleConnections(connections);

    for (int i = 0; i < 1000000; i++) {
      Activity activity = computeSampleActivity(connections);
    }

    float duration = (float)(clock() - start) / CLOCKS_PER_SEC;
    cout << "testTemporalMemoryUsage: " << duration << endl;
  }

  void ConnectionsPerformanceTest::setupSampleConnections(Connections &connections)
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

    presynapticCell.idx = 80;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 81;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 82;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    connections.updateSynapsePermanence(synapse, 0.15);

    segment = connections.createSegment(cell);

    presynapticCell.idx = 50;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 51;
    synapse = connections.createSynapse(segment, presynapticCell, 0.85);
    presynapticCell.idx = 52;
    synapse = connections.createSynapse(segment, presynapticCell, 0.15);
  }

  Activity ConnectionsPerformanceTest::computeSampleActivity(Connections &connections)
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

} // end namespace nupic

int main(int argc, char *argv[])
{
  ConnectionsPerformanceTest test = ConnectionsPerformanceTest();
  test.RunTests();
}