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
#include <nupic/algorithms/Anomaly.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/types/Types.hpp> // macro "UNUSED"

namespace testing {

using namespace std;
using namespace nupic;
using nupic::sdr::SDR;
using namespace nupic::algorithms::connections;
using ::nupic::algorithms::spatial_pooler::SpatialPooler;
using ::nupic::algorithms::temporal_memory::TemporalMemory;
using namespace nupic::algorithms::anomaly;

#define SEED 42

Random rng(SEED);

float runTemporalMemoryTest(UInt numColumns, UInt w,   int numSequences,
                                                       int numElements,
                                                       string label) {
  Timer timer(true);

  // Initialize

  TemporalMemory tm;
  tm.initialize( {numColumns} );

  cout << (float)timer.getElapsed() << " in " << label << ": initialize"  << endl;

  // generate data
  vector<vector<SDR>> sequences;
  for (int i = 0; i < numSequences; i++) {
    vector<SDR> sequence;
    SDR sdr({numColumns});
    for (int j = 0; j < numElements; j++) {
      const Real sparsity = w / static_cast<Real>(numColumns);
      sdr.randomize(sparsity, rng);
      sequence.push_back(sdr);
    }
    sequences.push_back(sequence);
  }

  // learn
  for (int i = 0; i < 5; i++) {
    for (auto sequence : sequences) {
      for (auto sdr : sequence) {
        tm.compute(sdr, true);
	//TODO get untrained anomaly score here
      }
      tm.reset();
    }
  }
  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn"  << endl;

  // test
  for (auto sequence : sequences) {
    for (auto sdr : sequence) {
      tm.compute(sdr, false);
      //TODO get trained (lower) anomaly
    }
    tm.reset();
  }
  //TODO check anomaly trained < anomaly untrained
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
    /* potentialRadius */               (numInputs + numColumns)
    );
  sp.setLocalAreaDensity(columnSparsity);

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



// TESTS
#if defined( NDEBUG) && !defined(NTA_OS_WINDOWS)
  const UInt COLS 	= 2048; //standard num of columns in SP/TM
  const UInt W 		= 50;
  const UInt SEQ 	= 50; //number of sequences ran in tests
  const UInt EPOCHS 	= 20; //tests run for epochs times
#else
  const UInt COLS 	= 20; //standard num of columns in SP/TM
  const UInt W 		= 3;
  const UInt SEQ 	= 25; //number of sequences ran in tests
  const UInt EPOCHS 	= 4; //only short in debug; is epochs/2 in some tests, that's why 4
#endif


/**
 * Tests typical usage of Connections with Temporal Memory.
 * format is: COLS, W(bits), EPOCHS, SEQUENCES
 */
TEST(ConnectionsPerformanceTest, testTM) {
	auto tim = runTemporalMemoryTest(COLS, W, EPOCHS, SEQ, "temporal memory");
#ifdef NDEBUG
	ASSERT_LE(tim, 2.5*Timer::getSpeed()); //there are times, we must be better. Bit underestimated for slow CI
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with a large Temporal Memory.
 */
TEST(ConnectionsPerformanceTest, testTMLarge) {
  auto tim = runTemporalMemoryTest(2*COLS, 6*W, EPOCHS/2, SEQ, "temporal memory (large)");
#ifdef NDEBUG
  ASSERT_LE(tim, 13*Timer::getSpeed());
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with Spatial Pooler.
 */
TEST(ConnectionsPerformanceTest, testSP) {
  auto tim = runSpatialPoolerTest(
    /* numInputs */          COLS,
    /* inputSparsity */      0.15f,
    /* numColumns */         COLS,
    /* columnSparsity */     0.05f,
    /* label */              "spatial pooler");

#ifdef NDEBUG
  ASSERT_LE(tim, 4.0f * Timer::getSpeed());
#endif
  UNUSED(tim);
}

} // end namespace
