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

#include <algorithm> // std::generate
#include <iostream>
#include <vector>

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/algorithms/Cells4.hpp"  //TODO use TM instead
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/encoders/ScalarEncoder.hpp"

#include "nupic/os/Timer.hpp"
#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/Random.hpp" 

namespace examples {

using namespace std;
using namespace nupic;
using namespace nupic::utils;
using nupic::ScalarEncoder;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::Cells4::Cells4;
using nupic::algorithms::anomaly::Anomaly;
using nupic::algorithms::anomaly::AnomalyMode;

// work-load 
void run() {
  const UInt COLS = 2048; // number of columns in SP, TP
  const UInt DIM_INPUT = 10000;
  const UInt CELLS = 10; // cells per column in TP
  const UInt EPOCHS = 5000; // number of iterations (calls to SP/TP compute() )
  std::cout << "starting test. DIM_INPUT=" << DIM_INPUT
  		<< ", DIM=" << COLS << ", CELLS=" << CELLS << std::endl;
  std::cout << "EPOCHS = " << EPOCHS << std::endl;


  // initialize SP, TP, Anomaly, AnomalyLikelihood
  Timer tInit(true);
  ScalarEncoder enc(133, -100.0, 100.0, DIM_INPUT, 0.0, 0.0, false);
  SpatialPooler sp(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS});
  Cells4 tp(COLS, CELLS, 12, 8, 15, 5, .5f, .8f, 1.0f, .1f, .1f, 0.0f,
            false, 42, true, false);
  Anomaly an(5, AnomalyMode::LIKELIHOOD);
  tInit.stop();

  // data for processing input
  vector<UInt> input(DIM_INPUT);
  vector<UInt> outSP(COLS); // active array, output of SP/TP
  vector<UInt> outTP(tp.nCells());
  vector<Real> rIn(COLS); // input for TP (must be Reals)
  vector<Real> rOut(tp.nCells());
  Real res = 0.0; //for anomaly:
  vector<UInt> prevPred_(outSP.size());
  Random rnd;
  
  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  Timer tAll(true);
  Timer tRng, tEnc, tSP, tTP, tAn;


  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    //Input
//    generate(input.begin(), input.end(), [&] () { return rnd.getUInt32(2); });
    tRng.start();
    const Real r = rnd.getUInt32(100) - rnd.getUInt32(100)*rnd.getReal64(); //rnd from range -100..100 
    tRng.stop();

    //Encode
    tEnc.start();
    enc.encodeIntoArray(r, input.data());
    tEnc.stop();

    //SP
    tSP.start();
    fill(outSP.begin(), outSP.end(), 0);
    sp.compute(input.data(), true, outSP.data());
    sp.stripUnlearnedColumns(outSP.data());
    tSP.stop();

    //TP
    tTP.start();
    rIn = VectorHelpers::castVectorType<UInt, Real>(outSP);
    tp.compute(rIn.data(), rOut.data(), true, true);
    outTP = VectorHelpers::castVectorType<Real, UInt>(rOut);
    tTP.stop();

    //Anomaly
    tAn.start();
    res = an.compute(outSP /*active*/, prevPred_ /*prev predicted*/); 
    prevPred_ = outTP; //to be used as predicted T-1
    tAn.stop();

    // print
    if (e == EPOCHS - 1) {
      tAll.stop();

      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << res << endl;
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(outSP), ",", "SP= ");
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(VectorHelpers::cellsToColumns(outTP, CELLS)), ",", "TP= ");
      NTA_CHECK(outSP[69] == 0) << "A value in SP computed incorrectly";
      NTA_CHECK(outTP[42] == 0) << "Incorrect value in TP";
      cout << "==============TIMERS============" << endl;
      cout << "Init:\t" << tInit.getElapsed() << endl;
      cout << "Random:\t" << tRng.getElapsed() << endl;
      cout << "Encode:\t" << tEnc.getElapsed() << endl;
      cout << "SP:\t" << tSP.getElapsed() << endl;
      cout << "TP:\t" << tTP.getElapsed() << endl;
      cout << "AN:\t" << tAn.getElapsed() << endl;

      const size_t timeTotal = tAll.getElapsed();
      cout << "Total elapsed time = " << timeTotal << " seconds" << endl;
      #ifdef NDEBUG
        const size_t CI_avg_time = 7*Timer::getSpeed(); //sec
        NTA_CHECK(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
          "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time;
      #endif
    }
  } //end for

} //end run()
} //-ns
