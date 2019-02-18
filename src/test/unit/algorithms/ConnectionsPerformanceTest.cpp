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

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/types/Types.hpp> // macro "UNUSED"

namespace testing {

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using ::nupic::algorithms::spatial_pooler::SpatialPooler;
using ::nupic::algorithms::temporal_memory::TemporalMemory;

#define SEED 42

Random rng(SEED);

std::vector<UInt32> _randomSDR(UInt n, UInt w);
void _feedTM(TemporalMemory &tm, vector<CellIdx> sdr, bool learn = true);

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
  return (float)timer.getElapsed();
}

float runSpatialPoolerTest(
                  UInt   numInputs,
                  Real   inputSparsity,
                  UInt   numColumns,
                  Real   columnSparsity,
                  string label)
{
#ifdef NDEBUG
  const auto trainTime = 1000u;
  const auto testTime  =  500u;
#else
  const auto trainTime = 10u;
  const auto testTime  =  5u;
#endif

  Timer timer;
  timer.start();

  // Initialize
  SpatialPooler sp(
    /* inputDimensions */               { numInputs },
    /* columnDimensions */              { numColumns },
    /* potentialRadius */               (numInputs + numColumns),
    /* potentialPct */                  0.5f,
    /* globalInhibition */              true,
    /* localAreaDensity */              columnSparsity,
    /* numActiveColumnsPerInhArea */    -1,
    /* stimulusThreshold */             6u,
    /* synPermInactiveDec */            0.01f,
    /* synPermActiveInc */              0.03f,
    /* synPermConnected */              0.4f,
    /* minPctOverlapDutyCycles */       0.001f,
    /* dutyCyclePeriod */               1000u,
    /* boostStrength */                 1.0f,
    /* seed */                          rng(),
    /* spVerbosity */                   0u,
    /* wrapAround */                    true);
  SDR input( sp.getInputDimensions() );
  SDR columns( sp.getColumnDimensions() );
  cout << (float)timer.getElapsed() << " in " << label << ": initialize"  << endl;

  // Learn
  for (auto i = 0u; i < trainTime; i++) {
    input.randomize( inputSparsity, rng );
    sp.compute( input, true, columns );
  }
  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn"  << endl;

  // Test
  for (auto i = 0u; i < testTime; i++) {
    input.randomize( inputSparsity, rng );
    sp.compute( input, false, columns );
  }
  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn + test"  << endl;
  timer.stop();
  return (float)timer.getElapsed();
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


// TESTS
#ifdef NDEBUG
  const UInt COLS = 2048; //standard num of columns in SP/TM
  const UInt SEQ = 50; //number of sequences ran in tests
  const UInt EPOCHS = 20; //tests run for epochs times
#else
  const UInt COLS = 20; //standard num of columns in SP/TM
  const UInt SEQ = 25; //number of sequences ran in tests
  const UInt EPOCHS = 4; //only short in debug; is epochs/2 in some tests, that's why 4
#endif


/**
 * Tests typical usage of Connections with Temporal Memory.
 * format is: COLS, W(bits), EPOCHS, SEQUENCES
 */
TEST(ConnectionsPerformanceTest, testTM) {
	auto tim = runTemporalMemoryTest(COLS, 40, EPOCHS, SEQ, "temporal memory");
#ifdef NDEBUG
	ASSERT_LE(tim, 1.0*Timer::getSpeed()); //there are times, we must be better. Bit underestimated for slow CI
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with a large Temporal Memory.
 */
TEST(ConnectionsPerformanceTest, testTMLarge) {
  auto tim = runTemporalMemoryTest(2*COLS, 328, EPOCHS/2, SEQ, "temporal memory (large)");
#ifdef NDEBUG
  ASSERT_LE(tim, 1.9*Timer::getSpeed());
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with Spatial Pooler.
 */
TEST(ConnectionsPerformanceTest, testSP) {
  auto tim = runSpatialPoolerTest(
    /* numInputs */          1024,
    /* inputSparsity */      0.15f,
    /* numColumns */         1024,
    /* columnSparsity */     0.05f,
    /* label */              "spatial pooler");

#ifdef NDEBUG
  ASSERT_LE(tim, 4.0f * Timer::getSpeed());
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with Temporal Pooler.
 */
TEST(ConnectionsPerformanceTest, testTP) {
  auto tim = runSpatialPoolerTest(
    /* numInputs */          4 * 1024,
    /* inputSparsity */      0.02f,
    /* numColumns */         1024 / 2,
    /* columnSparsity */     0.05f,
    /* label */              "temporal pooler");

#ifdef NDEBUG
  ASSERT_LE(tim, 4.0f * Timer::getSpeed());
#endif
  UNUSED(tim);
}

} // end namespace
