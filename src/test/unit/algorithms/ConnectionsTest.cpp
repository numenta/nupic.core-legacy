/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2014-2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Implementation of unit tests for Connections
 */

#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <htm/algorithms/Connections.hpp>

using namespace std;
using namespace htm;


void setupSampleConnections(Connections &connections) {
  // Cell with 1 segment.
  // Segment with:
  // - 1 connected synapse: active
  // - 2 matching synapses
  const Segment segment1_1 = connections.createSegment(10);
  connections.createSynapse(segment1_1, 150, 0.85f); //connected
  connections.createSynapse(segment1_1, 151, 0.15f);

  // Cell with 2 segments.
  // Segment with:
  // - 2 connected synapses: 2 active
  // - 3 matching synapses: 3 active
  const Segment segment2_1 = connections.createSegment(20, 2/* max number of segments per cell*/);
  connections.createSynapse(segment2_1, 80, 0.85f);
  connections.createSynapse(segment2_1, 81, 0.85f);
  Synapse synapse = connections.createSynapse(segment2_1, 82, 0.85f);
  connections.updateSynapsePermanence(synapse, 0.15f);

  // Segment with:
  // - 2 connected synapses: 1 active, 1 inactive
  // - 3 matching synapses: 2 active, 1 inactive
  // - 1 non-matching synapse: 1 active
  const Segment segment2_2 = connections.createSegment(20, 2);
  connections.createSynapse(segment2_2, 50, 0.85f);
  connections.createSynapse(segment2_2, 51, 0.85f);
  connections.createSynapse(segment2_2, 52, 0.15f);
  connections.createSynapse(segment2_2, 53, 0.05f);

  // Cell with one segment.
  // Segment with:
  // - 1 non-matching synapse: 1 active
  const Segment segment3_1 = connections.createSegment(30);
  connections.createSynapse(segment3_1, 53, 0.05f);
}

void computeSampleActivity(Connections &connections) {
  vector<UInt32> input = {50, 52, 53, 80, 81, 82, 150, 151};

  vector<SynapseIdx> numActiveConnectedSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  vector<SynapseIdx> numActivePotentialSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment,
                              numActivePotentialSynapsesForSegment, input);
}

/**
 * Creates a segment, and makes sure that it got created on the correct cell.
 */
TEST(ConnectionsTest, testCreateSegment) {
  Connections connections(1024);
  UInt32 cell = 10;

  Segment segment1 = connections.createSegment(cell);
  ASSERT_EQ(cell, connections.cellForSegment(segment1));

  Segment segment2 = connections.createSegment(cell);
  ASSERT_EQ(cell, connections.cellForSegment(segment2));

  vector<Segment> segments = connections.segmentsForCell(cell);
  ASSERT_EQ(segments.size(), 2ul);

  ASSERT_EQ(segment1, segments[0]);
  ASSERT_EQ(segment2, segments[1]);
}

/**
 * Creates a synapse, and makes sure that it got created on the correct
 * segment, and that its data was correctly stored.
 */
TEST(ConnectionsTest, testCreateSynapse) {
  Connections connections(1024);
  UInt32 cell = 10;
  Segment segment = connections.createSegment(cell);

  Synapse synapse1 = connections.createSynapse(segment, 50, 0.34f);
  ASSERT_EQ(segment, connections.segmentForSynapse(synapse1));

  Synapse synapse2 = connections.createSynapse(segment, 150, 0.48f);
  ASSERT_EQ(segment, connections.segmentForSynapse(synapse2));

  vector<Synapse> synapses = connections.synapsesForSegment(segment);
  ASSERT_EQ(synapses.size(), 2ul);

  ASSERT_EQ(synapse1, synapses[0]);
  ASSERT_EQ(synapse2, synapses[1]);

  SynapseData synapseData1 = connections.dataForSynapse(synapses[0]);
  ASSERT_EQ(50ul, synapseData1.presynapticCell);
  ASSERT_NEAR((Permanence)0.34, synapseData1.permanence, htm::Epsilon);

  SynapseData synapseData2 = connections.dataForSynapse(synapses[1]);
  ASSERT_EQ(synapseData2.presynapticCell, 150ul);
  ASSERT_NEAR((Permanence)0.48, synapseData2.permanence, htm::Epsilon);
}

/**
 * Creates a segment, destroys it, and makes sure it got destroyed along with
 * all of its synapses.
 */
TEST(ConnectionsTest, testDestroySegment) {
  Connections connections(1024);

  /*      segment1*/ connections.createSegment(10);
  Segment segment2 = connections.createSegment(20);
  /*      segment3*/ connections.createSegment(20);
  /*      segment4*/ connections.createSegment(30);

  connections.createSynapse(segment2, 80, 0.85f);
  connections.createSynapse(segment2, 81, 0.85f);
  connections.createSynapse(segment2, 82, 0.15f);

  ASSERT_EQ(4ul, connections.numSegments());
  ASSERT_EQ(3ul, connections.numSynapses());

  connections.destroySegment(segment2);

  ASSERT_EQ(3ul, connections.numSegments());
  ASSERT_EQ(0ul, connections.numSynapses());

  vector<SynapseIdx> numActiveConnectedSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  vector<SynapseIdx> numActivePotentialSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment,
                              numActivePotentialSynapsesForSegment,
                              {80, 81, 82});

  ASSERT_EQ(0ul, numActiveConnectedSynapsesForSegment[segment2]);
  ASSERT_EQ(0ul, numActivePotentialSynapsesForSegment[segment2]);
}

/**
 * Creates a segment, creates a number of synapses on it, destroys a synapse,
 * and makes sure it got destroyed.
 */
TEST(ConnectionsTest, testDestroySynapse) {
  Connections connections(1024);

  Segment segment = connections.createSegment(20);
  /*      synapse1*/ connections.createSynapse(segment, 80, 0.85f);
  Synapse synapse2 = connections.createSynapse(segment, 81, 0.85f);
  /*      synapse3*/ connections.createSynapse(segment, 82, 0.15f);

  ASSERT_EQ(3ul, connections.numSynapses());

  connections.destroySynapse(synapse2);

  ASSERT_EQ(2ul, connections.numSynapses());
  ASSERT_EQ(2ul, connections.synapsesForSegment(segment).size());

  vector<SynapseIdx> numActiveConnectedSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  vector<SynapseIdx> numActivePotentialSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment,
                              numActivePotentialSynapsesForSegment,
                              {80, 81, 82});

  ASSERT_EQ(1ul, numActiveConnectedSynapsesForSegment[segment]);
  ASSERT_EQ(2ul, numActivePotentialSynapsesForSegment[segment]);
}

/**
 * Creates segments and synapses, then destroys segments and synapses on
 * either side of them and verifies that existing Segment and Synapse
 * instances still point to the same segment / synapse as before.
 */
TEST(ConnectionsTest, PathsNotInvalidatedByOtherDestroys) {
  Connections connections(1024);

  Segment segment1 = connections.createSegment(11);
  /*      segment2*/ connections.createSegment(12);

  Segment segment3 = connections.createSegment(13);
  Synapse synapse1 = connections.createSynapse(segment3, 201, 0.85f);
  /*      synapse2*/ connections.createSynapse(segment3, 202, 0.85f);
  Synapse synapse3 = connections.createSynapse(segment3, 203, 0.85f);
  /*      synapse4*/ connections.createSynapse(segment3, 204, 0.85f);
  Synapse synapse5 = connections.createSynapse(segment3, 205, 0.85f);

  /*      segment4*/ connections.createSegment(14);
  Segment segment5 = connections.createSegment(15);

  ASSERT_EQ(203ul, connections.dataForSynapse(synapse3).presynapticCell);
  connections.destroySynapse(synapse1);
  EXPECT_EQ(203ul, connections.dataForSynapse(synapse3).presynapticCell);
  connections.destroySynapse(synapse5);
  EXPECT_EQ(203ul, connections.dataForSynapse(synapse3).presynapticCell);

  connections.destroySegment(segment1);
  EXPECT_EQ(3ul, connections.synapsesForSegment(segment3).size());
  connections.destroySegment(segment5);
  EXPECT_EQ(3ul, connections.synapsesForSegment(segment3).size());
  EXPECT_EQ(203ul, connections.dataForSynapse(synapse3).presynapticCell);
}

/**
 * Destroy a segment that has a destroyed synapse and a non-destroyed synapse.
 * Make sure nothing gets double-destroyed.
 */
TEST(ConnectionsTest, DestroySegmentWithDestroyedSynapses) {
  Connections connections(1024);

  Segment segment1 = connections.createSegment(11);
  Segment segment2 = connections.createSegment(12);

  /*      synapse1_1*/ connections.createSynapse(segment1, 101, 0.85f);
  Synapse synapse2_1 = connections.createSynapse(segment2, 201, 0.85f);
  /*      synapse2_2*/ connections.createSynapse(segment2, 202, 0.85f);

  ASSERT_EQ(3ul, connections.numSynapses());

  connections.destroySynapse(synapse2_1);

  ASSERT_EQ(2ul, connections.numSegments());
  ASSERT_EQ(2ul, connections.numSynapses());

  connections.destroySegment(segment2);

  EXPECT_EQ(1ul, connections.numSegments());
  EXPECT_EQ(1ul, connections.numSynapses());
}

/**
 * Destroy a segment that has a destroyed synapse and a non-destroyed synapse.
 * Create a new segment in the same place. Make sure its synapse count is
 * correct.
 */
TEST(ConnectionsTest, ReuseSegmentWithDestroyedSynapses) {
  Connections connections(1024);

  Segment segment = connections.createSegment(11);

  Synapse synapse1 = connections.createSynapse(segment, 201, 0.85f);
  /*      synapse2*/ connections.createSynapse(segment, 202, 0.85f);

  connections.destroySynapse(synapse1);

  ASSERT_EQ(1ul, connections.numSynapses(segment));

  connections.destroySegment(segment);
  Segment reincarnated = connections.createSegment(11);

  EXPECT_EQ(0ul, connections.numSynapses(reincarnated));
  EXPECT_EQ(0ul, connections.synapsesForSegment(reincarnated).size());
}

/**
 * Creates a synapse and updates its permanence, and makes sure that its
 * data was correctly updated.
 */
TEST(ConnectionsTest, testUpdateSynapsePermanence) {
  Connections connections(1024);
  Segment segment = connections.createSegment(10);
  Synapse synapse = connections.createSynapse(segment, 50, 0.34f);

  connections.updateSynapsePermanence(synapse, 0.21f);

  SynapseData synapseData = connections.dataForSynapse(synapse);
  ASSERT_NEAR(synapseData.permanence, (Real)0.21, htm::Epsilon);

  // Test permanence floor
  connections.updateSynapsePermanence(synapse, -0.02f);
  synapseData = connections.dataForSynapse(synapse);
  ASSERT_EQ(synapseData.permanence, (Real)0.0f );

  connections.updateSynapsePermanence(synapse, (Real)(-htm::Epsilon / 10.0));
  synapseData = connections.dataForSynapse(synapse);
  ASSERT_EQ(synapseData.permanence, (Real)0.0f );

  // Test permanence ceiling
  connections.updateSynapsePermanence(synapse, 1.02f);
  synapseData = connections.dataForSynapse(synapse);
  ASSERT_EQ(synapseData.permanence, (Real)1.0f );

  connections.updateSynapsePermanence(synapse, 1.0f + (Real)(htm::Epsilon / 10.0));
  synapseData = connections.dataForSynapse(synapse);
  ASSERT_EQ(synapseData.permanence, (Real)1.0f );
}

/**
 * Creates a sample set of connections, and makes sure that computing the
 * activity for a collection of cells with no activity returns the right
 * activity data.
 */
TEST(ConnectionsTest, testComputeActivity) {
  Connections connections(1024);

  // Cell with 1 segment.
  // Segment with:
  // - 1 connected synapse: active
  // - 2 matching synapses: active
  const Segment segment1_1 = connections.createSegment(10);
  connections.createSynapse(segment1_1, 150, 0.85f);
  connections.createSynapse(segment1_1, 151, 0.15f);

  // Cell with 1 segments.
  // Segment with:
  // - 2 connected synapses: 2 active
  // - 3 matching synapses: 3 active
  const Segment segment2_1 = connections.createSegment(20);
  connections.createSynapse(segment2_1, 80, 0.85f);
  connections.createSynapse(segment2_1, 81, 0.85f);
  Synapse synapse = connections.createSynapse(segment2_1, 82, 0.85f);
  connections.updateSynapsePermanence(synapse, 0.15f);

  vector<UInt32> input = {50, 52, 53, 80, 81, 82, 150, 151};

  vector<SynapseIdx> numActiveConnectedSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  vector<SynapseIdx> numActivePotentialSynapsesForSegment(
      connections.segmentFlatListLength(), 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment,
                              numActivePotentialSynapsesForSegment, input);

  ASSERT_EQ(1ul, numActiveConnectedSynapsesForSegment[segment1_1]);
  ASSERT_EQ(2ul, numActivePotentialSynapsesForSegment[segment1_1]);

  ASSERT_EQ(2ul, numActiveConnectedSynapsesForSegment[segment2_1]);
  ASSERT_EQ(3ul, numActivePotentialSynapsesForSegment[segment2_1]);
}

TEST(ConnectionsTest, testAdaptSynapses) {
  UInt numCells = 4;
  // NOTE: One segment per cell.
  UInt numInputs = 8;
  Connections con(numCells);

  vector<UInt> activeSegments;
  SDR input({numInputs});

  UInt potentialArr[4][8] =  {{1, 1, 1, 1, 0, 0, 0, 0},
                              {1, 0, 0, 0, 1, 1, 0, 1},
                              {0, 0, 1, 0, 0, 0, 1, 0},
                              {1, 0, 0, 0, 0, 0, 1, 0}};

  Real permanences[4][8] = {
      {0.200f, 0.120f, 0.090f, 0.060f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.150f, 0.000f, 0.000f, 0.000f, 0.180f, 0.120f, 0.000f, 0.450f},
      {0.000f, 0.000f, 0.004f, 0.000f, 0.000f, 0.000f, 0.910f, 0.000f},
      {0.070f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f}};

  Real truePerms[4][8] = {
      {0.300f, 0.110f, 0.080f, 0.160f, 0.000f, 0.000f, 0.000f, 0.000f},
      // Inc     Dec     Dec     Inc     -       -       -       -
      {0.250f, 0.000f, 0.000f, 0.000f, 0.280f, 0.110f, 0.000f, 0.440f},
      // Inc     -       -       -       Inc     Dec     -       Dec
      {0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f, 0.000f},
      // -       -      Floor    -      -        -     Ceiling   -
      {0.070f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f}};
      // -       -       -       -       -       -       -       -

  for (UInt cell = 0; cell < numCells; cell++) {
    Segment seg = con.createSegment(cell);
    for(UInt inp = 0; inp < numInputs; inp++) {
      if( potentialArr[cell][inp] )
        con.createSynapse(seg, inp, permanences[cell][inp]);
    }
  }

  input.setDense(SDR_dense_t({ 1, 0, 0, 1, 1, 0, 1, 0 }));
  activeSegments.assign({0, 1, 2});

  for(UInt seg : activeSegments)
    con.adaptSegment(seg, input, 0.1f, 0.01f);

  for (UInt cell = 0; cell < numCells; cell++) {
    vector<Real> perms( numInputs, 0.0f );
    for( Synapse syn : con.synapsesForSegment(cell) ) {
      auto synData = con.dataForSynapse( syn );
      perms[ synData.presynapticCell ] = synData.permanence;
    }
    for(UInt i = 0; i < numInputs; i++)
      ASSERT_NEAR( truePerms[cell][i], perms[i], htm::Epsilon );
  }
}

TEST(ConnectionsTest, testRaisePermanencesToThreshold) {
  UInt stimulusThreshold = 3;
  Real synPermConnected = 0.1f;
  UInt numInputs = 5;
  UInt numCells = 7;
  Connections con(numCells, synPermConnected);

  UInt potentialArr[7][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 0, 0, 1},
                             {0, 1, 1, 1, 0}};

  Real permArr[7][5] = {{0.000f, 0.110f,  0.095f, 0.092f, 0.010f},
                        {0.120f, 0.150f,  0.020f, 0.120f, 0.090f},
                        {0.510f, 0.081f,  0.025f, 0.089f, 0.310f},
                        {0.180f, 0.0601f, 0.110f, 0.011f, 0.030f},
                        {0.011f, 0.011f,  0.011f, 0.011f, 0.011f},
                        {0.120f, 0.056f,  0.000f, 0.000f, 0.078f},
                        {0.000f, 0.061f,  0.070f, 0.140f, 0.000f}};

  Real truePerm[7][5] = {
      {0.010f, 0.120f, 0.105f, 0.102f, 0.020f},    // incremented once
      {0.120f, 0.150f, 0.020f, 0.120f, 0.090f},    // no change
      {0.530f, 0.101f, 0.045f, 0.109f, 0.330f},    // increment twice
      {0.220f, 0.1001f,0.150f, 0.051f, 0.070f},    // increment four times
      {0.101f, 0.101f, 0.101f, 0.101f, 0.101f},    // increment 9 times
      {0.170f, 0.106f, 0.000f, 0.000f, 0.128f},    // increment 5 times
      {0.000f, 0.101f, 0.110f, 0.180f, 0.000f}};   // increment 4 times

  for (UInt i = 0; i < numCells; i++) {
    // Setup this cell / segment / synapses.
    con.createSegment(i);
    for (UInt j = 0; j < numInputs; j++) {
      if (potentialArr[i][j] > 0) {
        con.createSynapse( i, j, permArr[i][j] );
      }
    }
    // Run method under test.
    con.raisePermanencesToThreshold(i, stimulusThreshold);
    // Check results.
    for(auto syn : con.synapsesForSegment(i)) {
      auto synData = con.dataForSynapse( syn );
      UInt presyn  = synData.presynapticCell;
      ASSERT_NEAR(truePerm[i][presyn], synData.permanence, 0.01f);
    }
  }
 }


TEST(ConnectionsTest, testRaisePermanencesToThresholdOutOfBounds) {
  Connections con(1001, 0.666f);
 	
  // check empty segment (with no synapse data) 
  auto emptySegment = con.createSegment(0);
  auto synapses = con.synapsesForSegment(emptySegment);
  NTA_CHECK(synapses.empty()) << "We want to create a Segment with no synapses";
  EXPECT_NO_THROW( con.raisePermanencesToThreshold(emptySegment, 3u) )
    << "raisePermanence fails when empty Segment encountered";

  // check segment with 3 synapses, but wanted to raise 5
  auto segWith3Syn = con.createSegment(0);
  //add 3 synapses
  con.createSynapse( segWith3Syn, 33, 0.001f);
  con.createSynapse( segWith3Syn, 18, 0.25f);
  con.createSynapse( segWith3Syn, 121, 0.00001f);
  NTA_CHECK(con.synapsesForSegment(segWith3Syn).size() == 3) << "We failed to create 3 synapses on a segment";
  EXPECT_NO_THROW( con.raisePermanencesToThreshold(segWith3Syn, 5u) )
    << "raisePermanence fails when lower number of available synapses than requested by threshold";
}

TEST(ConnectionsTest, testSynapseCompetition) {

  struct testCase {
    UInt nsyn; // Total number of potential synapses on segment
    UInt ncon; // Number of connected synapses, before calling synapseCompetition
    UInt min;  // Bounds of synapseCompetition
    UInt max;  // Bounds of synapseCompetition
    // The target number of synapses can't be met, just make sure it does not crash.
    bool expect_fail = false;
  };

  testCase emptySegment;
  emptySegment.nsyn = 0;
  emptySegment.ncon = 0;
  emptySegment.min  = 3;
  emptySegment.max  = 100;
  emptySegment.expect_fail = true;

  testCase fullSegment;
  fullSegment.nsyn = 100;
  fullSegment.ncon = 100;
  fullSegment.min  = 3;
  fullSegment.max  = 100;

  testCase disconnect1;
  disconnect1.nsyn = 100;
  disconnect1.ncon = 100;
  disconnect1.min  = 3;
  disconnect1.max  = 99;

  testCase minimum;
  minimum.nsyn = 100;
  minimum.ncon = 5;
  minimum.min  = 10;
  minimum.max  = 30;

  testCase maximum;
  maximum.nsyn = 100;
  maximum.ncon = 77;
  maximum.min  = 10;
  maximum.max  = 30;

  testCase no_change1;
  no_change1.nsyn = 100;
  no_change1.ncon = 10;
  no_change1.min  = 10;
  no_change1.max  = 30;

  testCase no_change2;
  no_change2.nsyn = 100;
  no_change2.ncon = 20;
  no_change2.min  = 10;
  no_change2.max  = 30;

  testCase no_change3;
  no_change3.nsyn = 100;
  no_change3.ncon = 30;
  no_change3.min  = 10;
  no_change3.max  = 30;

  testCase exact1;
  exact1.nsyn = 100;
  exact1.ncon = 33;
  exact1.min  = 33;
  exact1.max  = 33;

  testCase exact2;
  exact2.nsyn = 100;
  exact2.ncon = 0;
  exact2.min  = 33;
  exact2.max  = 33;

  testCase exact3;
  exact3.nsyn = 100;
  exact3.ncon = 88;
  exact3.min  = 33;
  exact3.max  = 33;

  testCase corner1;
  corner1.nsyn = 100;
  corner1.ncon = 30;
  corner1.min  = 200;
  corner1.max  = 300;
  corner1.expect_fail = true;

  const Permanence thresh = 0.5f;
  Connections con(1u, thresh);
  Random rnd( 42u );
  CellIdx presyn = 0u;
  for(const testCase &test : {
          emptySegment, fullSegment, disconnect1, minimum, maximum, no_change1,
          no_change2, no_change3, exact1, exact2, exact3, corner1, })
  {
    const auto segment = con.createSegment( 0 );
    UInt ncon_done = 0;
    for(UInt i = test.nsyn; i > 0 ; --i) {
      // Randomly sample which synapses will connected.
      if( rnd.getReal64() <= Real64(test.ncon - ncon_done) / i ) {
        ncon_done++;
        con.createSynapse( segment, presyn++, rnd.realRange(thresh, 1.0f) );
      }
      else {
        con.createSynapse( segment, presyn++, rnd.realRange(0.0f, thresh) );
      }
    }
    // Check test setup is good.
    const auto &segData = con.dataForSegment( segment );
    ASSERT_EQ( test.nsyn, segData.synapses.size() );
    ASSERT_EQ( test.ncon, segData.numConnected );

    con.synapseCompetition( segment, test.min, test.max );

    // Check synapse data "numConnected" is accurate.
    int real_ncon = 0;
    for( const auto syn : segData.synapses ) {
      const auto &synData = con.dataForSynapse( syn );
      if( synData.permanence >= thresh - htm::Epsilon ) {
        real_ncon++;
      }
    }
    EXPECT_EQ( segData.numConnected, real_ncon );

    // Check results of synapse competition.
    if( not test.expect_fail ) {
      EXPECT_GE( segData.numConnected, test.min );
      EXPECT_LE( segData.numConnected, test.max );
      if( test.ncon >= test.min and test.ncon <= test.max ) {
        EXPECT_EQ( segData.numConnected, test.ncon );
      }
    }
  }
}

TEST(ConnectionsTest, testBumpSegment) {
  UInt numInputs = 8;
  UInt numSegments = 5;
  Connections con(1);

  UInt potentialArr[5][8] = {{1, 1, 1, 1, 0, 0, 0, 0},
                             {1, 0, 0, 0, 1, 1, 0, 1},
                             {0, 0, 1, 0, 1, 1, 1, 0},
                             {1, 1, 1, 0, 0, 0, 1, 0},
                             {1, 1, 1, 1, 1, 1, 1, 1}};

  Real permArr[5][8] = {
      {0.200f, 0.120f, 0.090f, 0.040f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.150f, 0.000f, 0.000f, 0.000f, 0.180f, 0.120f, 0.000f, 0.450f},
      {0.000f, 0.000f, 0.074f, 0.000f, 0.062f, 0.054f, 0.110f, 0.000f},
      {0.051f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.178f, 0.000f},
      {0.100f, 0.738f, 0.085f, 0.002f, 0.052f, 0.008f, 0.208f, 0.034f}};

  Real deltaArr[5] = {0.010f, 0.750f, 0.000f, -0.001f, -0.010f};

  Real truePermArr[5][8] = {
      {0.210f, 0.130f, 0.100f, 0.050f, 0.000f, 0.000f, 0.000f, 0.000f},
      {0.900f, 0.000f, 0.000f, 0.000f, 0.930f, 0.870f, 0.000f, 1.000f},
      {0.000f, 0.000f, 0.074f, 0.000f, 0.062f, 0.054f, 0.110f, 0.000f}, // unchanged
      {0.050f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.177f, 0.000f},
      {0.090f, 0.728f, 0.075f, 0.000f, 0.042f, 0.000f, 0.198f, 0.024f}};

  for (UInt seg = 0; seg < numSegments; seg++) {
    auto segment = con.createSegment(0);
    for(UInt i = 0; i < numInputs; i++)
      if( potentialArr[seg][i] )
        con.createSynapse(segment, i, permArr[seg][i]);

    con.bumpSegment( segment, deltaArr[seg] );

    for(auto synapse : con.synapsesForSegment(segment)) {
      auto synData = con.dataForSynapse( synapse );
      auto presyn  = synData.presynapticCell;
      ASSERT_FLOAT_EQ( synData.permanence, truePermArr[seg][presyn] );
    }
  }
}

/**
 * Test the mapSegmentsToCells method.
 */
TEST(ConnectionsTest, testMapSegmentsToCells) {
  Connections connections(1024);

  const Segment segment1 = connections.createSegment(42);
  const Segment segment2 = connections.createSegment(42);
  const Segment segment3 = connections.createSegment(43);

  const vector<Segment> segments = {segment1, segment2, segment3, segment1};
  vector<CellIdx> cells(segments.size());

  connections.mapSegmentsToCells(
      segments.data(), segments.data() + segments.size(), cells.data());

  const vector<CellIdx> expected = {42, 42, 43, 42};
  ASSERT_EQ(expected, cells);
}

bool TEST_EVENT_HANDLER_DESTRUCTED = false;

class TestConnectionsEventHandler : public ConnectionsEventHandler {
public:
  TestConnectionsEventHandler()
      : didCreateSegment(false), didDestroySegment(false),
        didCreateSynapse(false), didDestroySynapse(false),
        didUpdateSynapsePermanence(false) {}

  virtual ~TestConnectionsEventHandler() {
    TEST_EVENT_HANDLER_DESTRUCTED = true;
  }

  virtual void onCreateSegment(Segment segment) { didCreateSegment = true; }

  virtual void onDestroySegment(Segment segment) { didDestroySegment = true; }

  virtual void onCreateSynapse(Synapse synapse) { didCreateSynapse = true; }

  virtual void onDestroySynapse(Synapse synapse) { didDestroySynapse = true; }

  virtual void onUpdateSynapsePermanence(Synapse synapse,
                                         Permanence permanence) {
    didUpdateSynapsePermanence = true;
  }

  bool didCreateSegment;
  bool didDestroySegment;
  bool didCreateSynapse;
  bool didDestroySynapse;
  bool didUpdateSynapsePermanence;
};

/**
 * Make sure each event handler gets called.
 */
TEST(ConnectionsTest, subscribe) {
  Connections connections(1024, 0.5f);

  TestConnectionsEventHandler *handler = new TestConnectionsEventHandler();
  auto token = connections.subscribe(handler);

  ASSERT_FALSE(handler->didCreateSegment);
  Segment segment = connections.createSegment(42);
  EXPECT_TRUE(handler->didCreateSegment);

  ASSERT_FALSE(handler->didCreateSynapse);
  Synapse synapse = connections.createSynapse(segment, 41, 0.25f);
  EXPECT_TRUE(handler->didCreateSynapse);

  ASSERT_FALSE(handler->didUpdateSynapsePermanence);
  connections.updateSynapsePermanence(synapse, 0.60f);
  EXPECT_TRUE(handler->didUpdateSynapsePermanence);

  ASSERT_FALSE(handler->didDestroySynapse);
  connections.destroySynapse(synapse);
  EXPECT_TRUE(handler->didDestroySynapse);

  ASSERT_FALSE(handler->didDestroySegment);
  connections.destroySegment(segment);
  EXPECT_TRUE(handler->didDestroySegment);

  connections.unsubscribe(token);
}

/**
 * Make sure the event handler is destructed on unsubscribe.
 */
TEST(ConnectionsTest, unsubscribe) {
  Connections connections(1024);
  TestConnectionsEventHandler *handler = new TestConnectionsEventHandler();
  auto token = connections.subscribe(handler);

  TEST_EVENT_HANDLER_DESTRUCTED = false;
  connections.unsubscribe(token);
  EXPECT_TRUE(TEST_EVENT_HANDLER_DESTRUCTED);
}

/**
 * Creates a sample set of connections, and makes sure that we can get the
 * correct number of segments.
 */
TEST(ConnectionsTest, testNumSegments) {
  Connections connections(1024);
  setupSampleConnections(connections);

  ASSERT_EQ(4ul, connections.numSegments());
}

/**
 * Creates a sample set of connections, and makes sure that we can get the
 * correct number of synapses.
 */
TEST(ConnectionsTest, testNumSynapses) {
  Connections connections(1024);
  setupSampleConnections(connections);

  ASSERT_EQ(10ul, connections.numSynapses());
}

/**
 * Creates a sample set of connections with destroyed segments/synapses,
 * computes sample activity, and makes sure that we can save to a
 * filestream and load it back correctly.
 */
TEST(ConnectionsTest, testSaveLoad) {
  Connections c1(1024), c2;
  setupSampleConnections(c1);

  auto segment = c1.createSegment(10);

  c1.createSynapse(segment, 400, 0.5);
  c1.destroySegment(segment);

  computeSampleActivity(c1);

  {
    stringstream ss;
    c1.save(ss);
    c2.load(ss);
  }

  ASSERT_EQ(c1, c2);
}

TEST(ConnectionsTest, testCreateSegmentOverflow) {
    const auto LIMIT = std::numeric_limits<Segment>::max();
    if(LIMIT <= 256) { //connections::Segment is too large (likely uint32), so this test would run, but memory 
      // would kill the machine! 
      // to test this test and the code works OK, change connections::Segment to unsigned char
      //TODO use GTEST_SKIP() when we can have gtest > 1.8.1 to skip at runtime

    Connections c(1024);
    size_t i = 0;
    for(i=0; i < LIMIT; i++) {
      EXPECT_NO_THROW(c.createSegment(0));
    }
    EXPECT_ANY_THROW(c.createSegment(0)) << "num segments on cell c0 " << (size_t)c.numSegments(0) 
	    << " total num segs: " << (size_t)c.numSegments() << "data-type limit " << LIMIT;
  }
}

TEST(ConnectionsTest, testCreateSynapseOverflow) {
  const auto LIMIT = std::numeric_limits<Synapse>::max();
  if(LIMIT <= 256) { //connections::Synapse is too large (likely uint32), so this test would run, but memory
    // would kill the machine!
    // to test this test and the code works OK, change connections::Synapse to unsigned char
    //TODO use GTEST_SKIP() when we can have gtest > 1.8.1 to skip at runtime
    Connections c(1024);
    const Segment seg = c.createSegment(0);

    size_t i = 0;
    for(i=0; i < LIMIT; i++) {
      EXPECT_NO_THROW(c.createSynapse(seg, (CellIdx)99, (Permanence)0.1337));
    }
    EXPECT_ANY_THROW(c.createSynapse(seg, (CellIdx)99, (Permanence)0.1337)) << "num synapses on segment s0 " << (size_t)c.numSynapses(seg)
      << " total num syns: " << (size_t)c.numSynapses() << "data-type limit " << LIMIT;
  }
}

TEST(ConnectionsTest, testTimeseries) {
  Connections C( 1, .5, true );
  auto seg = C.createSegment(0);
  SDR presyn({ 10u });
  for( UInt cell = 0; cell < presyn.size; cell++ ) {
    C.createSynapse(seg, cell, 0.5f );
  }
  // Use the same SDR many times.  Hold the segment active & learning.
  presyn.randomize( 0.5f );
  vector<SynapseIdx> output( 1u );
  for( int i = 0; i < 10; i++ ) {
    C.computeActivity( output, presyn.getSparse() );
    C.adaptSegment( seg, presyn, 0.1f, 0.1f );
  }
  // Check that the synapse permanences did not saturate.
  for( const auto syn : C.synapsesForSegment( seg ) ) {
    const auto &synData = C.dataForSynapse( syn );
    ASSERT_GT( synData.permanence, 0.01f );
    ASSERT_LT( synData.permanence, 0.99f );
  }
  // Retry this test except call Connections::reset() before every cycle, which
  // effectively turns off the timeseries parameter.
  for( int i = 0; i < 10; i++ ) {
    C.reset();
    C.computeActivity( output, presyn.getSparse() );
    C.adaptSegment( seg, presyn, 0.1f, 0.1f );
  }
  // Check that the synapse permanences staturated.  This is the failure
  // condition which the timeseries modifications prevent.
  for( const auto syn : C.synapsesForSegment( seg ) ) {
    const auto &synData = C.dataForSynapse( syn );
    ASSERT_TRUE( (synData.permanence == 0.0f) or (synData.permanence == 1.0f) );
  }
}
