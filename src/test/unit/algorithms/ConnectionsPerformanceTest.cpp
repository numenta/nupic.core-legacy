/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

#include "gtest/gtest.h"

/** @file
 * Implementation of performance tests for Connections
 */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

#include "ConnectionsPerformanceTest.hpp"

namespace testing { 

using namespace std;
using namespace nupic;
using nupic::algorithms::connections::Segment;

#define SEED 42

float ConnectionsPerformanceTest::runTemporalMemoryTest(UInt numColumns, UInt w,
                                                       int numSequences,
                                                       int numElements,
                                                       string label) {
  clock_t timer = clock();

  // Initialize

  TemporalMemory tm;
  vector<UInt> columnDim;
  columnDim.push_back(numColumns);
  tm.initialize(columnDim);

  checkpoint(timer, label + ": initialize");

  // Learn

  vector<vector<vector<CellIdx>>> sequences;
  vector<vector<CellIdx>> sequence;
  vector<CellIdx> sdr;

  for (int i = 0; i < numSequences; i++) {
    for (int j = 0; j < numElements; j++) {
      sdr = randomSDR(numColumns, w);
      sequence.push_back(sdr);
    }

    sequences.push_back(sequence);
  }

  for (int i = 0; i < 5; i++) {
    for (auto sequence : sequences) {
      for (auto sdr : sequence) {
        feedTM(tm, sdr);
        tm.reset();
      }
    }
  }

  checkpoint(timer, label + ": initialize + learn");

  // Test

  for (auto sequence : sequences) {
    for (auto sdr : sequence) {
      feedTM(tm, sdr, false);
      tm.reset();
    }
  }

  const float totalTime = checkpoint(timer, label + ": initialize + learn + test");
  return totalTime;
}

float ConnectionsPerformanceTest::runSpatialPoolerTest(UInt numCells,
                                                      UInt numInputs, UInt w,
                                                      UInt numWinners,
                                                      string label) {
  clock_t timer = clock();

  Connections connections(numCells);
  Segment segment;
  vector<CellIdx> sdr;

  // Initialize

  for (UInt c = 0; c < numCells; c++) {
    segment = connections.createSegment(c);

    for (UInt i = 0; i < numInputs; i++) {
      const Permanence permanence =
          max((Permanence)0.000001, (Permanence)rand() / RAND_MAX);
      connections.createSynapse(segment, i, permanence);
    }
  }

  checkpoint(timer, label + ": initialize");

  // Learn

  vector<CellIdx> winnerCells;
  Permanence permanence;

  for (int i = 0; i < 500; i++) {
    sdr = randomSDR(numInputs, w);
    vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                numActivePotentialSynapsesForSegment, sdr, 0.5);
    winnerCells = computeSPWinnerCells(connections, numWinners,
                                       numActiveConnectedSynapsesForSegment);

    for (CellIdx winnerCell : winnerCells) {
      segment = connections.getSegment(winnerCell, 0);

      const vector<Synapse> &synapses = connections.synapsesForSegment(segment);

      for (SynapseIdx i = 0; i < (SynapseIdx)synapses.size();) {
        const Synapse synapse = synapses[i];
        const SynapseData &synapseData = connections.dataForSynapse(synapse);
        permanence = synapseData.permanence;

        if (find(sdr.begin(), sdr.end(), synapseData.presynapticCell) !=
            sdr.end()) {
          permanence += 0.2;
        } else {
          permanence -= 0.1;
        }

        permanence = max(permanence, (Permanence)0);
        permanence = min(permanence, (Permanence)1);

        if (permanence == 0) {
          connections.destroySynapse(synapse);
          // The synapses list is updated in-place, so don't update `i`.
        } else {
          connections.updateSynapsePermanence(synapse, permanence);
          i++;
        }
      }
    }
  }

  checkpoint(timer, label + ": initialize + learn");

  // Test

  for (int i = 0; i < 500; i++) {
    sdr = randomSDR(numInputs, w);
    vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                numActivePotentialSynapsesForSegment, sdr, 0.5);
    winnerCells = computeSPWinnerCells(connections, numWinners,
                                       numActiveConnectedSynapsesForSegment);
  }

  const float totalTime = checkpoint(timer, label + ": initialize + learn + test");
  return totalTime;
}


float ConnectionsPerformanceTest::checkpoint(clock_t timer, string text) {
  const float duration = (float)(clock() - timer) / CLOCKS_PER_SEC;
  cout << duration << " in " << text << endl;
  return duration;
}


vector<CellIdx> ConnectionsPerformanceTest::randomSDR(UInt n, UInt w) {
  set<UInt> sdrSet = set<UInt>();
  vector<CellIdx> sdr;

  for (UInt i = 0; i < w; i++) {
    sdrSet.insert(rand() % (UInt)n); //TODO use our Random
  }

  for (UInt c : sdrSet) {
    sdr.push_back(c);
  }

  return sdr;
}


void ConnectionsPerformanceTest::feedTM(TemporalMemory &tm, vector<CellIdx> sdr,
                                        bool learn) {
  vector<UInt> activeColumns;

  for (auto c : sdr) {
    activeColumns.push_back(c);
  }

  tm.compute(activeColumns.size(), activeColumns.data(), learn);
}


vector<CellIdx> ConnectionsPerformanceTest::computeSPWinnerCells(
    Connections &connections, UInt numCells,
    const vector<UInt> &numActiveSynapsesForSegment) {
  // Activate every segment, then choose the top few.
  vector<Segment> activeSegments;
  for (Segment segment = 0; segment < numActiveSynapsesForSegment.size();
       segment++) {
    activeSegments.push_back(segment);
  }

  set<CellIdx> winnerCells;
  std::sort(
      activeSegments.begin(), activeSegments.end(), [&](Segment a, Segment b) {
        return numActiveSynapsesForSegment[a] > numActiveSynapsesForSegment[b];
      });

  for (Segment segment : activeSegments) {
    winnerCells.insert(connections.cellForSegment(segment));
    if (winnerCells.size() >= numCells) {
      break;
    }
  }

  return vector<CellIdx>(winnerCells.begin(), winnerCells.end());
}



// TESTS
ConnectionsPerformanceTest t;
const UInt SEQ = 100; //number of sequences ran in tests
const UInt EPOCHS = 20; //epochs tests run
const UInt COLS = 2048; //standard num of columns in SP/TM

void SetUp() {
  srand(SEED);
  t = testing::ConnectionsPerformanceTest();
}

/**
 * Tests typical usage of Connections with Temporal Memory.
 * format is: COLS, W(bits), EPOCHS, SEQUENCES
 */
TEST(ConnectionsPerformanceTest, testTM) {
	auto tim = t.runTemporalMemoryTest(COLS, 40, EPOCHS, SEQ, "temporal memory");
	ASSERT_LE(tim, 3.5f); //there are times, we must be better. Bit underestimated for slow CI
}

/**
 * Tests typical usage of Connections with a large Temporal Memory.
 */
TEST(ConnectionsPerformanceTest, testTMLarge) {
  auto tim = t.runTemporalMemoryTest(2*COLS, 328, 10, SEQ, "temporal memory (large)");
  ASSERT_LE(tim, 7.0f);
}

/**
 * Tests typical usage of Connections with Spatial Pooler.
 */
TEST(ConnectionsPerformanceTest, testSP) {
  auto tim = t.runSpatialPoolerTest(COLS, COLS, EPOCHS, SEQ, "spatial pooler");
  ASSERT_LE(tim, 10.0f);
}

/**
 * Tests typical usage of Connections with Temporal Pooler.
 */
TEST(ConnectionsPerformanceTest, testTP) {
  auto tim = t.runSpatialPoolerTest(COLS, 16384, 10, SEQ, "temporal pooler");
  ASSERT_LE(tim, 80.0f);
}

} // end namespace 
