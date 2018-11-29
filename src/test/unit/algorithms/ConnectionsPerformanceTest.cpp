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
#include <cmath> //for sin

#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/os/Timer.hpp>

namespace testing {

using namespace std;
using namespace nupic;
using ::nupic::algorithms::connections::Segment;
using ::nupic::algorithms::temporal_memory::TemporalMemory;

#define SEED 42

Random rng(SEED);

std::vector<UInt32> _randomSDR(UInt n, UInt w);
void _feedTM(TemporalMemory &tm, vector<CellIdx> sdr, bool learn = true);
std::vector<CellIdx> _computeSPWinnerCells(Connections &connections, UInt numCells,
                       const vector<UInt> &numActiveSynapsesForSegment);

float runTemporalMemoryTest(UInt numColumns, UInt w,   int numSequences,
                                                       int numElements,
                                                       string label) {
  Timer timer(true);

  // Initialize

  TemporalMemory tm;
  vector<UInt> columnDim;
  columnDim.push_back(numColumns);
  tm.initialize(columnDim);

  cout << (float)timer.getElapsed() << " in " << label << ": initialize"  << endl;

  // Learn

  vector<vector<vector<CellIdx>>> sequences;
  vector<vector<CellIdx>> sequence;
  vector<CellIdx> sdr;

  for (int i = 0; i < numSequences; i++) {
    for (int j = 0; j < numElements; j++) {
      sdr = _randomSDR(numColumns, w);
      sequence.push_back(sdr);
    }

    sequences.push_back(sequence);
  }

  for (int i = 0; i < 5; i++) {
    for (auto sequence : sequences) {
      for (auto sdr : sequence) {
        _feedTM(tm, sdr);
        tm.reset();
      }
    }
  }

  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn"  << endl;

  // Test

  for (auto sequence : sequences) {
    for (auto sdr : sequence) {
      _feedTM(tm, sdr, false);
      tm.reset();
    }
  }

  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn + test"  << endl;
  timer.stop();
  return timer.getElapsed();
}

float runSpatialPoolerTest(UInt numCells, UInt numInputs, UInt w,
                           UInt numWinners, string label) {
  Timer timer;
  timer.start();

  Connections connections(numCells);
  Segment segment;
  vector<CellIdx> sdr;

  // Initialize

  for (UInt c = 0; c < numCells; c++) {
    segment = connections.createSegment(c);

    for (UInt i = 0; i < numInputs; i++) {
      const Permanence permanence = (Permanence)rng.getReal64();
      connections.createSynapse(segment, i, permanence);
    }
  }

  cout << (float)timer.getElapsed() << " in " << label << ": initialize"  << endl;

  // Learn

  vector<CellIdx> winnerCells;
  Permanence permanence;

  for (int i = 0; i < 500; i++) {
    sdr = _randomSDR(numInputs, w);
    vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                numActivePotentialSynapsesForSegment, sdr, 0.5);
    winnerCells = _computeSPWinnerCells(connections, numWinners,
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

  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn"  << endl;

  // Test

  for (int i = 0; i < 500; i++) {
    sdr = _randomSDR(numInputs, w);
    vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
    connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                numActivePotentialSynapsesForSegment, sdr, 0.5);
    winnerCells = _computeSPWinnerCells(connections, numWinners,
                                       numActiveConnectedSynapsesForSegment);
  }

  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn + test"  << endl;
  timer.stop();
  return timer.getElapsed();
}


vector<CellIdx> _randomSDR(UInt n, UInt w) {
  set<UInt> sdrSet = set<UInt>();
  vector<CellIdx> sdr;

  for (UInt i = 0; i < w; i++) {
    sdrSet.insert(rng.getUInt32(n));
  }

  for (UInt c : sdrSet) {
    sdr.push_back(c);
  }

  return sdr;
}


void _feedTM(TemporalMemory &tm, vector<CellIdx> sdr, bool learn) {
  vector<UInt> activeColumns;

  for (auto c : sdr) {
    activeColumns.push_back(c);
  }

  tm.compute(activeColumns.size(), activeColumns.data(), learn);
}


vector<CellIdx> _computeSPWinnerCells(Connections &connections, UInt numCells,
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

float _SPEED = -1;
/**
 * estimate speed (CPU & load) of the current system.
 * Tests must perform relative to this value
 */
float getSpeed() {
  if (_SPEED == -1) {
    Timer t(true);
    //this code just wastes CPU time to estimate speed
    vector<Real> data(10000000);
    for(Size i=0; i<data.size(); i++) {
      data[i]=(Real)rng.getUInt32(80085);
      auto t = data[i];
      data[i] = data[data.size()-i];
      data[data.size()-i]=t;
    }
    rng.shuffle(begin(data), end(data));
    vector<Real> sins;
    for (auto d : data) {
      sins.push_back(sin(d)/cos(d));
    }
    data = rng.sample<Real>(sins, 666);
    NTA_CHECK(data.size() == 666);
    t.stop();
    _SPEED = max(1.0, t.getElapsed());

  }
  return _SPEED;
}


// TESTS
const UInt SEQ = 100; //number of sequences ran in tests
#ifdef NDEBUG
  const UInt EPOCHS = 20; //only short in debug; is epochs/2 in some tests, that's why 4
#else
  const UInt EPOCHS = 4; //epochs tests run
#endif
const UInt COLS = 2048; //standard num of columns in SP/TM


/**
 * Tests typical usage of Connections with Temporal Memory.
 * format is: COLS, W(bits), EPOCHS, SEQUENCES
 */
TEST(ConnectionsPerformanceTest, testTM) {
	auto tim = runTemporalMemoryTest(COLS, 40, EPOCHS, SEQ, "temporal memory");
	ASSERT_LE(tim, 2.0*getSpeed()); //there are times, we must be better. Bit underestimated for slow CI
}

/**
 * Tests typical usage of Connections with a large Temporal Memory.
 */
TEST(ConnectionsPerformanceTest, testTMLarge) {
  auto tim = runTemporalMemoryTest(2*COLS, 328, EPOCHS/2, SEQ, "temporal memory (large)");
  ASSERT_LE(tim, 3.8*getSpeed());
}

/**
 * Tests typical usage of Connections with Spatial Pooler.
 */
TEST(ConnectionsPerformanceTest, testSP) {
  auto tim = runSpatialPoolerTest(COLS, COLS, EPOCHS, SEQ, "spatial pooler");
  ASSERT_LE(tim, 6.3*getSpeed());
}

/**
 * Tests typical usage of Connections with Temporal Pooler.
 */
TEST(ConnectionsPerformanceTest, testTP) {
  auto tim = runSpatialPoolerTest(COLS, 16384, EPOCHS/2, SEQ/50, "temporal pooler");
  ASSERT_LE(tim, 13.0*getSpeed());
}

} // end namespace
