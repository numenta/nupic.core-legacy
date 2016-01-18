/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for Connections
 */

#include <fstream>
#include <iostream>
#include <nupic/algorithms/Connections.hpp>
#include "gtest/gtest.h"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;

#define EPSILON 0.0000001

namespace {

  void setupSampleConnections(Connections &connections)
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

  Activity computeSampleActivity(Connections &connections)
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

  /**
   * Creates a segment, and makes sure that it got created on the correct cell.
   */
  TEST(ConnectionsTest, testCreateSegment)
  {
    Connections connections(1024);
    Segment segment;
    Cell cell(10);

    segment = connections.createSegment(cell);
    ASSERT_EQ(segment.idx, 0);
    ASSERT_EQ(segment.cell.idx, cell.idx);

    segment = connections.createSegment(cell);
    ASSERT_EQ(segment.idx, 1);
    ASSERT_EQ(segment.cell.idx, cell.idx);

    vector<Segment> segments = connections.segmentsForCell(cell);
    ASSERT_EQ(segments.size(), 2);

    for (SegmentIdx i = 0; i < (SegmentIdx)segments.size(); i++) {
      ASSERT_EQ(segments[i].idx, i);
      ASSERT_EQ(segments[i].cell.idx, cell.idx);
    }
  }

  /**
   * Creates many segments on a cell, until hits segment limit. Then creates
   * another segment, and checks that it destroyed the least recently used
   * segment and created a new one in its place.
   */
  TEST(ConnectionsTest, testCreateSegmentReuse)
  {
    Connections connections(1024, 2);
    Cell cell;
    Segment segment;
    vector<Segment> segments;

    setupSampleConnections(connections);

    auto numSegments = connections.numSegments();
    Activity activity = computeSampleActivity(connections);

    cell.idx = 20;

    segment = connections.createSegment(cell);
    // Should have reused segment with index 1
    ASSERT_EQ(segment.idx, 1);

    segments = connections.segmentsForCell(cell);
    ASSERT_EQ(segments.size(), 2);

    ASSERT_EQ(numSegments, connections.numSegments());

    segment = connections.createSegment(cell);
    // Should have reused segment with index 0
    ASSERT_EQ(segment.idx, 0);

    segments = connections.segmentsForCell(cell);
    ASSERT_EQ(segments.size(), 2);

    ASSERT_EQ(numSegments, connections.numSegments());
  }

  /**
   * Creates a synapse, and makes sure that it got created on the correct
   * segment, and that its data was correctly stored.
   */
  TEST(ConnectionsTest, testCreateSynapse)
  {
    Connections connections(1024);
    Cell cell(10), presynapticCell;
    Segment segment = connections.createSegment(cell);
    Synapse synapse;

    presynapticCell.idx = 50;
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);
    ASSERT_EQ(synapse.idx, 0);
    ASSERT_EQ(synapse.segment.idx, segment.idx);

    presynapticCell.idx = 150;
    synapse = connections.createSynapse(segment, presynapticCell, 0.48);
    ASSERT_EQ(synapse.idx, 1);
    ASSERT_EQ(synapse.segment.idx, segment.idx);

    vector<Synapse> synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 2);

    for (SynapseIdx i = 0; i < synapses.size(); i++) {
      ASSERT_EQ(synapses[i].idx, i);
      ASSERT_EQ(synapses[i].segment.idx, segment.idx);
      ASSERT_EQ(synapses[i].segment.cell.idx, cell.idx);
    }

    SynapseData synapseData;

    synapseData = connections.dataForSynapse(synapses[0]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 50);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.34, EPSILON);

    synapseData = connections.dataForSynapse(synapses[1]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 150);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.48, EPSILON);
  }

  /**
  * Creates a synapse over the synapses per segment limit, and verifies
  * that the lowest permanence synapse is removed to make room for the new
  * synapse.
  */
  TEST(ConnectionsTest, testSynapseReuse)
  {
    // Limit to only two synapses per segment
    Connections connections(1024, 1024, 2);
    Cell cell(10), presynapticCell;
    Segment segment = connections.createSegment(cell);
    Synapse synapse;
    vector<Synapse> synapses;
    SynapseData synapseData;

    presynapticCell.idx = 50;
    synapse = connections.createSynapse(segment, presynapticCell, 0.34);
    presynapticCell.idx = 51;
    synapse = connections.createSynapse(segment, presynapticCell, 0.48);

    // Verify that the synapses we added are there
    synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 2);
    synapseData = connections.dataForSynapse(synapses[0]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 50);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.34, EPSILON);
    synapseData = connections.dataForSynapse(synapses[1]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 51);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.48, EPSILON);

    // Add an additional synapse over the limit
    presynapticCell.idx = 52;
    synapse = connections.createSynapse(segment, presynapticCell, 0.52);

    // Verify that the lowest permanence synapse was removed
    synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 2);
    synapseData = connections.dataForSynapse(synapses[0]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 51);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.48, EPSILON);
    synapseData = connections.dataForSynapse(synapses[1]);
    ASSERT_EQ(synapseData.presynapticCell.idx, 52);
    ASSERT_NEAR(synapseData.permanence, (Permanence)0.52, EPSILON);
  }

  /**
   * Creates a segment, destroys it, and makes sure it got destroyed along with
   * all of its synapses.
   */
  TEST(ConnectionsTest, testDestroySegment)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    auto numSegments = connections.numSegments();

    cell.idx = 20;
    segment.cell = cell;
    segment.idx = 0;
    connections.destroySegment(segment);

    ASSERT_EQ(connections.numSegments(), numSegments-1);
    ASSERT_THROW(connections.synapsesForSegment(segment);, runtime_error);

    Activity activity = computeSampleActivity(connections);

    ASSERT_EQ(activity.activeSegmentsForCell.size(), 0);
    ASSERT_EQ(activity.numActiveSynapsesForSegment.size(), 2);
  }

  /**
   * Creates a segment, creates a number of synapses on it, destroys a synapse,
   * and makes sure it got destroyed.
   */
  TEST(ConnectionsTest, testDestroySynapse)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;
    Synapse synapse;

    setupSampleConnections(connections);
    auto numSynapses = connections.numSynapses();

    cell.idx = 20;
    segment.cell = cell;
    segment.idx = 0;
    synapse.segment = segment;
    synapse.idx = 0;
    connections.destroySynapse(synapse);

    ASSERT_EQ(connections.numSynapses(), numSynapses-1);
    vector<Synapse> synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 2);

    Activity activity = computeSampleActivity(connections);

    ASSERT_EQ(activity.activeSegmentsForCell.size(), 0);

    segment.cell.idx = 20; segment.idx = 0;
    ASSERT_EQ(activity.numActiveSynapsesForSegment[segment], 1);
  }

  /**
   * Creates a synapse and updates its permanence, and makes sure that its
   * data was correctly updated.
   */
  TEST(ConnectionsTest, testUpdateSynapsePermanence)
  {
    Connections connections(1024);
    Cell cell(10), presynapticCell(50);
    Segment segment = connections.createSegment(cell);
    Synapse synapse = connections.createSynapse(segment, presynapticCell, 0.34);

    connections.updateSynapsePermanence(synapse, 0.21);

    SynapseData synapseData = connections.dataForSynapse(synapse);
    ASSERT_NEAR(synapseData.permanence, (Real)0.21, EPSILON);
  }

  /**
   * Creates a sample set of connections, and makes sure that getting the most
   * active segment for a collection of cells returns the right segment.
   */
  TEST(ConnectionsTest, testMostActiveSegmentForCells)
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

    ASSERT_EQ(result, true);

    ASSERT_EQ(segment.cell.idx, 20);
    ASSERT_EQ(segment.idx, 0);
  }

  /**
   * Creates a sample set of connections, and makes sure that getting the most
   * active segment for a collection of cells with no activity returns
   * no segment.
   */
  TEST(ConnectionsTest, testMostActiveSegmentForCellsNone)
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

    ASSERT_EQ(result, false);
  }

  /**
   * Creates a sample set of connections, computes some activity for it,
   * and checks that we can get the correct least recently used segment
   * for a number of cells.
   *
   * Then, destroys the least recently used segment, computes more activity,
   * creates another segment, and checks that the least recently used segment
   * is not the newly created one.
   */
  TEST(ConnectionsTest, testLeastRecentlyUsedSegment)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);

    cell.idx = 5;
    ASSERT_EQ(connections.leastRecentlyUsedSegment(cell, segment), false);

    cell.idx = 20;

    ASSERT_EQ(connections.leastRecentlyUsedSegment(cell, segment), true);
    ASSERT_EQ(segment.idx, 0);

    computeSampleActivity(connections);

    ASSERT_EQ(connections.leastRecentlyUsedSegment(cell, segment), true);
    ASSERT_EQ(segment.idx, 1);

    connections.destroySegment(segment);

    ASSERT_EQ(connections.leastRecentlyUsedSegment(cell, segment), true);
    ASSERT_EQ(segment.idx, 0);

    computeSampleActivity(connections);

    segment = connections.createSegment(cell);

    ASSERT_EQ(connections.leastRecentlyUsedSegment(cell, segment), true);
    ASSERT_EQ(segment.idx, 0);
  }

  /**
   * Creates a sample set of connections, and makes sure that computing the
   * activity for a collection of cells with no activity returns the right
   * activity data.
   */
  TEST(ConnectionsTest, testComputeActivity)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    ASSERT_EQ(activity.activeSegmentsForCell.size(), 1);
    cell.idx = 20;
    ASSERT_EQ(activity.activeSegmentsForCell[cell].size(), 1);
    segment = activity.activeSegmentsForCell[cell][0];
    ASSERT_EQ(segment.idx, 0);
    ASSERT_EQ(segment.cell.idx, 20);

    ASSERT_EQ(activity.numActiveSynapsesForSegment.size(), 3);
    segment.cell.idx = 10; segment.idx = 0;
    ASSERT_EQ(activity.numActiveSynapsesForSegment[segment], 1);
    segment.cell.idx = 20; segment.idx = 0;
    ASSERT_EQ(activity.numActiveSynapsesForSegment[segment], 2);
    segment.cell.idx = 20; segment.idx = 1;
    ASSERT_EQ(activity.numActiveSynapsesForSegment[segment], 1);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * active segments from the computed activity.
   */
  TEST(ConnectionsTest, testActiveSegments)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    vector<Segment> activeSegments = connections.activeSegments(activity);

    ASSERT_EQ(activeSegments.size(), 1);
    segment = activeSegments[0];
    ASSERT_EQ(segment.idx, 0);
    ASSERT_EQ(segment.cell.idx, 20);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * active cells from the computed activity.
   */
  TEST(ConnectionsTest, testActiveCells)
  {
    Connections connections(1024);
    Cell cell;
    Segment segment;

    setupSampleConnections(connections);
    Activity activity = computeSampleActivity(connections);

    vector<Cell> activeCells = connections.activeCells(activity);

    ASSERT_EQ(activeCells.size(), 1);
    ASSERT_EQ(activeCells[0].idx, 20);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * correct number of segments.
   */
  TEST(ConnectionsTest, testNumSegments)
  {
    Connections connections(1024);
    setupSampleConnections(connections);

    ASSERT_EQ(connections.numSegments(), 3);
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * correct number of synapses.
   */
  TEST(ConnectionsTest, testNumSynapses)
  {
    Connections connections(1024);
    setupSampleConnections(connections);

    ASSERT_EQ(connections.numSynapses(), 8);
  }

  /**
   * Creates a sample set of connections with destroyed segments/synapses,
   * computes sample activity, and makes sure that we can write to a
   * filestream and read it back correctly.
   */
  TEST(ConnectionsTest, testWriteRead)
  {
    const char* filename = "ConnectionsSerialization.tmp";
    Connections c1(1024, 1024, 1024), c2;
    setupSampleConnections(c1);

    Segment segment;
    Cell cell, presynapticCell;

    cell.idx = 10;
    presynapticCell.idx = 400;
    segment = c1.createSegment(cell);
    c1.createSynapse(segment, presynapticCell, 0.5);
    c1.destroySegment(segment);

    computeSampleActivity(c1);

    ofstream os(filename, ios::binary);
    c1.write(os);
    os.close();

    ifstream is(filename, ios::binary);
    c2.read(is);
    is.close();

    ASSERT_EQ(c1, c2);

    int ret = ::remove(filename);
    NTA_CHECK(ret == 0) << "Failed to delete " << filename;
  }

  TEST(ConnectionsTest, testSaveLoad)
  {
    Connections c1(1024, 1024, 1024), c2;
    setupSampleConnections(c1);

    Cell cell(10), presynapticCell(400);
    auto segment = c1.createSegment(cell);

    c1.createSynapse(segment, presynapticCell, 0.5);
    c1.destroySegment(segment);

    computeSampleActivity(c1);

    {
      stringstream ss;
      c1.save(ss);
      c2.load(ss);
    }

    ASSERT_EQ(c1, c2);
  }

} // end namespace nupic
