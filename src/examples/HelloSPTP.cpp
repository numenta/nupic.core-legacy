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

#include "nupic/algorithms/Cells4.hpp"
#include "nupic/algorithms/BacktrackingTMCpp.hpp"
#include "nupic/algorithms/TemporalMemory.hpp"

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

using TP =     nupic::algorithms::Cells4::Cells4;
using BackTM = nupic::algorithms::backtracking_tm::BacktrackingTMCpp;
using TM =     nupic::algorithms::temporal_memory::TemporalMemory;

using nupic::algorithms::anomaly::Anomaly;
using nupic::algorithms::anomaly::AnomalyMode;

// work-load
void run(UInt EPOCHS = 5000) {
  const UInt COLS = 2048; // number of columns in SP, TP
  const UInt DIM_INPUT = 10000;
  const UInt CELLS = 10; // cells per column in TP
#ifndef NDEBUG
  EPOCHS = 2; // make test faster in Debug
#endif

  std::cout << "starting test. DIM_INPUT=" << DIM_INPUT
  		<< ", DIM=" << COLS << ", CELLS=" << CELLS << std::endl;
  std::cout << "EPOCHS = " << EPOCHS << std::endl;


  // initialize SP, TP, Anomaly, AnomalyLikelihood
  Timer tInit(true);
  ScalarEncoder enc(133, -100.0, 100.0, DIM_INPUT, 0.0, 0.0, false);
  NTA_INFO << "SP (l) local inhibition is slow, so we reduce its data 10x smaller"; //to make it reasonably fast for test, for comparison x10
  SpatialPooler spGlobal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler spLocal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS/10u}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);

  TP tp(COLS, CELLS, 12, 8, 15, 5, .5f, .8f, 1.0f, .1f, .1f, 0.0f,
            false, 42, true, false);
  BackTM backTM(COLS, CELLS); //TODO get all, described parameters
  TM tm(vector<UInt>{COLS}, CELLS);

  Anomaly an(5, AnomalyMode::PURE);
  Anomaly anLikelihood(5, AnomalyMode::LIKELIHOOD);
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
  Timer tRng, tEnc, tSPloc, tSPglob, tTP, tBackTM, tTM, 
	tAn, tAnLikelihood;


  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    //Input
//    generate(input.begin(), input.end(), [&] () { return rnd.getUInt32(2); });
    tRng.start();
    const Real r = (Real)(rnd.getUInt32(100) - rnd.getUInt32(100)*rnd.getReal64()); //rnd from range -100..100
    tRng.stop();

    //Encode
    tEnc.start();
    enc.encodeIntoArray(r, input.data());
    tEnc.stop();

    //SP (global x local) 
    tSPloc.start();
    fill(outSP.begin(), outSP.end(), 0);
    spLocal.compute(input.data(), true, outSP.data());
    spLocal.stripUnlearnedColumns(outSP.data());
    tSPloc.stop();

    tSPglob.start();
    fill(outSP.begin(), outSP.end(), 0);
    spGlobal.compute(input.data(), true, outSP.data());
    spGlobal.stripUnlearnedColumns(outSP.data());
    vector<UInt> outSPsparse = VectorHelpers::binaryToSparse(outSP);
    tSPglob.stop();


    //TP (TP x BackTM x TM)
    tTP.start();
    rIn = VectorHelpers::castVectorType<UInt, Real>(outSP);
    tp.compute(rIn.data(), rOut.data(), true, true);
    outTP = VectorHelpers::castVectorType<Real, UInt>(rOut);
    tTP.stop();

    tBackTM.start();
    backTM.compute(rIn.data(), true /*learn*/, true /*infer*/);
    const auto backAct = backTM.getActiveState();
    const auto backPred = backTM.getPredictedState();
    const vector<char> vAct(backAct, backAct + backTM.getNumCells());
    const vector<char> bPred(backPred, backPred + backTM.getNumCells());
    tBackTM.stop();

    tTM.start();
    tm.compute(outSPsparse.size(), outSPsparse.data(), true /*learn*/);
    const auto tmAct = tm.getActiveCells();
    tm.activateDendrites(); //must be called before getPredictiveCells 
    const auto tmPred = tm.getPredictiveCells();
    //TODO assert tmAct == spOut
    //TODO merge Act + Pred and use for anomaly from TM
    tTM.stop();
 

    //Anomaly (pure x likelihood)
    tAn.start();
    res = an.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAn.stop();

    tAnLikelihood.start();
    anLikelihood.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAnLikelihood.stop();

    prevPred_ = outTP; //to be used as predicted T-1

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
      cout << "SP (l):\t" << tSPloc.getElapsed() << "(x10)" << endl;
      cout << "SP (g):\t" << tSPglob.getElapsed() << endl;
      cout << "TP:\t" << tTP.getElapsed() << endl;
      cout << "TM:\t" << tTM.getElapsed() << endl;
      cout << "BackTM:\t" << tBackTM.getElapsed() << endl;
      cout << "AN:\t" << tAn.getElapsed() << endl;
      cout << "AN:\t" << tAnLikelihood.getElapsed() << endl;

      const size_t timeTotal = (size_t)floor(tAll.getElapsed());
      cout << "Total elapsed time = " << timeTotal << " seconds" << endl;
      if(EPOCHS >= 100) { //show only relevant values, ie don't run in valgrind (ndebug, epochs=5) run
#ifdef _MSC_VER
          const size_t CI_avg_time = (size_t)floor(30*Timer::getSpeed()); //sec
#else
          const size_t CI_avg_time = (size_t)floor(7*Timer::getSpeed()); //sec
#endif
        NTA_CHECK(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
          "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time;
      }
    }
  } //end for

} //end run()
} //-ns
