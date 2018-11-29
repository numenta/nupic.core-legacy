/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <algorithm> // std::generate
#include <ctime>     // std::time
#include <iostream>
#include <vector>

#include "nupic/algorithms/Cells4.hpp"  //TODO use TM instead
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/encoders/ScalarEncoder.hpp"

#include "nupic/os/Timer.hpp"
#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/Random.hpp" 

namespace testing { 

using namespace std;
using namespace nupic;
using namespace nupic::utils;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::Cells4::Cells4;

TEST(HelloSPTPTest, performance) {

  const UInt COLS = 2048; // number of columns in SP, TP
  const UInt DIM_INPUT = 10000;
  const UInt CELLS = 8; // cells per column in TP
#ifdef NDEBUG
  const UInt EPOCHS = 1000; // number of iterations (calls to SP/TP compute() )
#else
  const UInt EPOCHS = 3; //run only short test in debug
#endif
  std::cout << "starting test. DIM_INPUT=" << DIM_INPUT
  		<< ", DIM=" << COLS << ", CELLS=" << CELLS << std::endl;
  std::cout << "EPOCHS = " << EPOCHS << std::endl;

  // generate random input
  ScalarEncoder enc(1337/*w*/, -1000.0, 1000.1, (int)DIM_INPUT/*n*/, 0.0, 0.0, false);
  vector<UInt> outSP(COLS); // active array, output of SP/TP

  // initialize SP, TP
  SpatialPooler sp(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS});
  Cells4 tp(COLS, CELLS, 12, 8, 15, 5, .5f, .8f, 1.0f, .1f, .1f, 0.0f,
            false, 42, true, false);

  vector<UInt> outTP(tp.nCells());
  vector<Real> rIn(COLS); // input for TP (must be Reals)
  vector<Real> rOut(tp.nCells());
  Random rnd;

  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  Timer stopwatch(true);


  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    const Real val = rnd.getUInt32(1000)-(rnd.getUInt32(1000)*rnd.getReal64());
    const auto input = enc.encode(val); 
    fill(outSP.begin(), outSP.end(), 0);
    EXPECT_NO_THROW(sp.compute(input.data(), true, outSP.data()));
    sp.stripUnlearnedColumns(outSP.data());

    rIn = VectorHelpers::castVectorType<UInt, Real>(outSP);
    EXPECT_NO_THROW(tp.compute(rIn.data(), rOut.data(), true, true));
    outTP = VectorHelpers::castVectorType<Real, UInt>(rOut);

    // print
    if (e == EPOCHS - 1) {
      cout << "Epoch = " << e << endl;
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(outSP), ",", "SP= ");
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(VectorHelpers::cellsToColumns(outTP, CELLS)), ",", "TP= ");
      ASSERT_EQ(outSP[69], 0) << "A value in SP computed incorrectly";
      ASSERT_EQ(outTP[42], 0) << "Incorrect value in TP";
    }
  }

  stopwatch.stop();
  const size_t timeTotal = stopwatch.getElapsed();
  const size_t CI_avg_time = 45; //sec
  cout << "Total elapsed time = " << timeTotal << " seconds" << endl;
  EXPECT_TRUE(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
	  "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time;

}
} //ns
