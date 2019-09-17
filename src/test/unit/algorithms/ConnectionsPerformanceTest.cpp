/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2015-2016, Numenta, Inc.
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

#include "gtest/gtest.h"

/** @file
 * Implementation of performance tests for Connections
 */

#include <fstream>
#include <iostream>

#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/algorithms/TemporalMemory.hpp>
#include <htm/utils/Random.hpp>
#include <htm/os/Timer.hpp>
#include <htm/types/Types.hpp> // macro "UNUSED"
#include <htm/utils/MovingAverage.hpp>

namespace testing {

using namespace std;
using namespace htm;

#define SEED 42

Random rng(SEED);

float runTemporalMemoryTest(UInt numColumns, UInt w,   int numSequences, //TODO rather than learning large/small TM, test on large sequence vs many small seqs
                                                       int numElements,
                                                       string label) {
  Timer timer(true);
  MovingAverage anom10(numSequences * numElements); //used for averaging anomaly scores
  Real avgAnomBefore = 1.0f, avgAnomAfter = 1.0f;
  NTA_CHECK(avgAnomBefore >= avgAnomAfter) << "TM should lear and avg anomalies improve, but we got: "
	  << avgAnomBefore << " and now: " << avgAnomAfter; //invariant

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
    for (const auto& sequence : sequences) {
      for (const auto& sdr : sequence) {
        tm.compute(sdr, true);
	const Real an = tm.anomaly;
	avgAnomAfter = anom10.compute(an); //average anomaly score
      }
      tm.reset();
    }
    NTA_CHECK(avgAnomBefore >= avgAnomAfter) << "TM should learn and avg anomalies improve, but we got: "
      << avgAnomBefore << " and now: " << avgAnomAfter; //invariant
    avgAnomBefore = avgAnomAfter; //update
  }
  cout << (float)timer.getElapsed() << " in " << label << ": initialize + learn"  << endl;

  // test
  for (auto sequence : sequences) {
    for (auto sdr : sequence) {
      tm.compute(sdr, false);
      avgAnomAfter = anom10.compute(tm.anomaly);
    }
    tm.reset();
  }

#if defined NDEBUG && !defined(NTA_OS_WINDOWS) //because Win & Debug run shorter training due to time, so learning is not as good
  NTA_CHECK(avgAnomAfter <= 0.021f) << "Anomaly scores diverged: "<< avgAnomAfter;
#endif
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
	ASSERT_LE(tim, 3.3f*Timer::getSpeed()); //there are times, we must be better. Bit underestimated for slow CI
#endif
  UNUSED(tim);
}

/**
 * Tests typical usage of Connections with a large Temporal Memory.
 */
TEST(ConnectionsPerformanceTest, testTMLarge) {
  auto tim = runTemporalMemoryTest(2*COLS, 6*W, EPOCHS/2, SEQ, "temporal memory (large)");
#ifdef NDEBUG
  ASSERT_LE(tim, 15*Timer::getSpeed());
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
