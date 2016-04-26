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
    // Cell with 1 segment.
    // Segment with:
    // - 1 active synapse
    // - 2 matching synapses
    const Segment segment1_1 = connections.createSegment(Cell(10));
    connections.createSynapse(segment1_1, Cell(150), 0.85);
    connections.createSynapse(segment1_1, Cell(151), 0.15);

    // Cell with 2 segments.
    // Segment with:
    // - 2 active synapses
    // - 3 matching synapses
    const Segment segment2_1 = connections.createSegment(Cell(20));
    connections.createSynapse(segment2_1, Cell(80), 0.85);
    connections.createSynapse(segment2_1, Cell(81), 0.85);
    Synapse synapse = connections.createSynapse(segment2_1, Cell(82), 0.85);
    connections.updateSynapsePermanence(synapse, 0.15);

    // Segment with:
    // - 2 active synapses (1 inactive)
    // - 3 matching synapses (1 inactive)
    // - 1 non-matching synapse
    const Segment segment2_2 = connections.createSegment(Cell(20));
    connections.createSynapse(segment2_2, Cell(50), 0.85);
    connections.createSynapse(segment2_2, Cell(51), 0.85);
    connections.createSynapse(segment2_2, Cell(52), 0.15);
    connections.createSynapse(segment2_2, Cell(53), 0.05);

    // Cell with one segment.
    // Segment with:
    // - 1 non-matching synapse
    const Segment segment3_1 = connections.createSegment(Cell(30));
    connections.createSynapse(segment3_1, Cell(53), 0.05);
  }

  Activity computeSampleActivity(Connections &connections)
  {
    vector<Cell> input;

    input.push_back(Cell(150));
    input.push_back(Cell(151));
    input.push_back(Cell(50));
    input.push_back(Cell(52));
    input.push_back(Cell(53));
    input.push_back(Cell(80));
    input.push_back(Cell(81));
    input.push_back(Cell(82));

    Activity activity = connections.computeActivity(input,
                                                    0.50, 2,
                                                    0.10, 1);
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
    EXPECT_EQ(0, synapse.idx);

    // Verify that the lowest permanence synapse was removed
    synapses = connections.synapsesForSegment(segment);
    ASSERT_EQ(synapses.size(), 2);
    synapseData = connections.dataForSynapse(synapses[0]);
    ASSERT_EQ(52, synapseData.presynapticCell.idx);
    ASSERT_NEAR((Permanence)0.52, synapseData.permanence, EPSILON);
    synapseData = connections.dataForSynapse(synapses[1]);
    ASSERT_EQ(51, synapseData.presynapticCell.idx);
    ASSERT_NEAR((Permanence)0.48, synapseData.permanence, EPSILON);
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

    UInt32 numSegmentsWithActivesynapses = 0;
    for (UInt32 numActiveSynapses : activity.numActiveSynapsesForSegment)
    {
      if (numActiveSynapses > 0)
      {
        numSegmentsWithActivesynapses++;
      }
    }
    ASSERT_EQ(2, numSegmentsWithActivesynapses);
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

    ASSERT_EQ(1, activity.numActiveSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(20))).flatIdx
                ]);
  }

  /**
   * Creates segments and synapses, then destroys segments and synapses on
   * either side of them and verifies that existing Segment and Synapse
   * instances still point to the same segment / synapse as before.
   */
  TEST(ConnectionsTest, PathsNotInvalidatedByOtherDestroys)
  {
    Connections connections(1024);

    Segment segment1 = connections.createSegment(Cell(11));
    /*      segment2*/ connections.createSegment(Cell(12));

    Segment segment3 = connections.createSegment(Cell(13));
    Synapse synapse1 = connections.createSynapse(segment3, Cell(201), 0.85);
    /*      synapse2*/ connections.createSynapse(segment3, Cell(202), 0.85);
    Synapse synapse3 = connections.createSynapse(segment3, Cell(203), 0.85);
    /*      synapse4*/ connections.createSynapse(segment3, Cell(204), 0.85);
    Synapse synapse5 = connections.createSynapse(segment3, Cell(205), 0.85);

    /*      segment4*/ connections.createSegment(Cell(14));
    Segment segment5 = connections.createSegment(Cell(15));

    ASSERT_EQ(203, connections.dataForSynapse(synapse3).presynapticCell.idx);
    connections.destroySynapse(synapse1);
    EXPECT_EQ(203, connections.dataForSynapse(synapse3).presynapticCell.idx);
    connections.destroySynapse(synapse5);
    EXPECT_EQ(203, connections.dataForSynapse(synapse3).presynapticCell.idx);

    connections.destroySegment(segment1);
    EXPECT_EQ(3, connections.synapsesForSegment(segment3).size());
    connections.destroySegment(segment5);
    EXPECT_EQ(3, connections.synapsesForSegment(segment3).size());
    EXPECT_EQ(203, connections.dataForSynapse(synapse3).presynapticCell.idx);
  }

  /**
   * Destroy a segment that has a destroyed synapse and a non-destroyed synapse.
   * Make sure nothing gets double-destroyed.
   */
  TEST(ConnectionsTest, DestroySegmentWithDestroyedSynapses)
  {
    Connections connections(1024);

    Segment segment1 = connections.createSegment(Cell(11));
    Segment segment2 = connections.createSegment(Cell(12));

    /*      synapse1_1*/ connections.createSynapse(segment1, Cell(101), 0.85);
    Synapse synapse2_1 = connections.createSynapse(segment2, Cell(201), 0.85);
    /*      synapse2_2*/ connections.createSynapse(segment2, Cell(202), 0.85);

    ASSERT_EQ(3, connections.numSynapses());

    connections.destroySynapse(synapse2_1);

    ASSERT_EQ(2, connections.numSegments());
    ASSERT_EQ(2, connections.numSynapses());

    connections.destroySegment(segment2);

    EXPECT_EQ(1, connections.numSegments());
    EXPECT_EQ(1, connections.numSynapses());
  }

  /**
   * Destroy a segment that has a destroyed synapse and a non-destroyed synapse.
   * Create a new segment in the same place. Make sure its synapse count is
   * correct.
   */
  TEST(ConnectionsTest, ReuseSegmentWithDestroyedSynapses)
  {
    Connections connections(1024);

    Segment segment = connections.createSegment(Cell(11));

    Synapse synapse1 = connections.createSynapse(segment, Cell(201), 0.85);
    /*      synapse2*/ connections.createSynapse(segment, Cell(202), 0.85);

    connections.destroySynapse(synapse1);

    ASSERT_EQ(1, connections.numSynapses(segment));

    connections.destroySegment(segment);
    Segment reincarnated = connections.createSegment(Cell(11));

    EXPECT_EQ(0, connections.numSynapses(reincarnated));
    EXPECT_EQ(0, connections.synapsesForSegment(reincarnated).size());
  }

  /**
   * Destroy some segments then verify that the maxSegmentsPerCell is still
   * correctly applied.
   */
  TEST(ConnectionsTest, DestroySegmentsThenReachLimit)
  {
    Connections connections(1024, 2, 2);

    {
      Segment segment1 = connections.createSegment(Cell(11));
      Segment segment2 = connections.createSegment(Cell(11));
      ASSERT_EQ(2, connections.numSegments());
      connections.destroySegment(segment1);
      connections.destroySegment(segment2);
      ASSERT_EQ(0, connections.numSegments());
    }

    {
      connections.createSegment(Cell(11));
      EXPECT_EQ(1, connections.numSegments());
      connections.createSegment(Cell(11));
      EXPECT_EQ(2, connections.numSegments());
      Segment segment = connections.createSegment(Cell(11));
      EXPECT_LT(segment.idx, 2);
      EXPECT_EQ(2, connections.numSegments());
    }
  }

  /**
   * Destroy some synapses then verify that the maxSynapsesPerSegment is still
   * correctly applied.
   */
  TEST(ConnectionsTest, DestroySynapsesThenReachLimit)
  {
    Connections connections(1024, 2, 2);

    Segment segment = connections.createSegment(Cell(10));

    {
      Synapse synapse1 = connections.createSynapse(segment, Cell(201), 0.85);
      Synapse synapse2 = connections.createSynapse(segment, Cell(202), 0.85);
      ASSERT_EQ(2, connections.numSynapses());
      connections.destroySynapse(synapse1);
      connections.destroySynapse(synapse2);
      ASSERT_EQ(0, connections.numSynapses());
    }

    {
      connections.createSynapse(segment, Cell(201), 0.85);
      EXPECT_EQ(1, connections.numSynapses());
      connections.createSynapse(segment, Cell(202), 0.90);
      EXPECT_EQ(2, connections.numSynapses());
      Synapse synapse = connections.createSynapse(segment, Cell(203), 0.80);
      EXPECT_LT(synapse.idx, 2);
      EXPECT_EQ(2, connections.numSynapses());
    }
  }

  /**
   * Hit the maxSegmentsPerCell threshold multiple times. Make sure it works
   * more than once.
   */
  TEST(ConnectionsTest, ReachSegmentLimitMultipleTimes)
  {
    Connections connections(1024, 2, 2);

    connections.createSegment(Cell(10));
    ASSERT_EQ(1, connections.numSegments());
    connections.createSegment(Cell(10));
    ASSERT_EQ(2, connections.numSegments());
    connections.createSegment(Cell(10));
    ASSERT_EQ(2, connections.numSegments());
    connections.createSegment(Cell(10));
    EXPECT_EQ(2, connections.numSegments());
  }

  /**
   * Hit the maxSynapsesPerSegment threshold multiple times. Make sure it works
   * more than once.
   */
  TEST(ConnectionsTest, ReachSynapseLimitMultipleTimes)
  {
    Connections connections(1024, 2, 2);

    Segment segment = connections.createSegment(Cell(10));
    connections.createSynapse(segment, Cell(201), 0.85);
    ASSERT_EQ(1, connections.numSynapses());
    connections.createSynapse(segment, Cell(202), 0.90);
    ASSERT_EQ(2, connections.numSynapses());
    connections.createSynapse(segment, Cell(203), 0.80);
    ASSERT_EQ(2, connections.numSynapses());
    Synapse synapse = connections.createSynapse(segment, Cell(204), 0.80);
    EXPECT_LT(synapse.idx, 2);
    EXPECT_EQ(2, connections.numSynapses());
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

    ASSERT_EQ(activity.numActiveSynapsesForSegment.size(), 4);
    ASSERT_EQ(activity.numMatchingSynapsesForSegment.size(), 4);
    ASSERT_EQ(1,
              activity.numActiveSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(10))).flatIdx
                ]);
    ASSERT_EQ(2,
              activity.numActiveSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(20))).flatIdx
                ]);
    ASSERT_EQ(3,
              activity.numMatchingSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(20))).flatIdx
                ]);
    ASSERT_EQ(1,
              activity.numActiveSynapsesForSegment[
                connections.dataForSegment(Segment(1, Cell(20))).flatIdx
                ]);
    ASSERT_EQ(2,
              activity.numMatchingSynapsesForSegment[
                connections.dataForSegment(Segment(1, Cell(20))).flatIdx
                ]);
    ASSERT_EQ(0,
              activity.numActiveSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(30))).flatIdx
                ]);
    ASSERT_EQ(0,
              activity.numMatchingSynapsesForSegment[
                connections.dataForSegment(Segment(0, Cell(30))).flatIdx
                ]);
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

    ASSERT_EQ(4, connections.numSegments());
  }

  /**
   * Creates a sample set of connections, and makes sure that we can get the
   * correct number of synapses.
   */
  TEST(ConnectionsTest, testNumSynapses)
  {
    Connections connections(1024);
    setupSampleConnections(connections);

    ASSERT_EQ(10, connections.numSynapses());
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
