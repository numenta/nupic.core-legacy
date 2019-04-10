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

#include "HelloSPTP.hpp"

#include "nupic/types/Sdr.hpp"

#include "nupic/algorithms/Anomaly.hpp"

#include "nupic/algorithms/TemporalMemory.hpp"

#include "nupic/algorithms/SpatialPooler.hpp"

#include "nupic/encoders/ScalarEncoder.hpp"

#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/Random.hpp"

namespace examples {

using namespace std;
using namespace nupic;
using namespace nupic::utils;

using nupic::sdr::SDR;
using nupic::encoders::ScalarEncoder;
using nupic::encoders::ScalarEncoderParameters;

using nupic::algorithms::spatial_pooler::SpatialPooler;

using TM =     nupic::algorithms::temporal_memory::TemporalMemory;

using nupic::algorithms::anomaly::Anomaly;
using nupic::algorithms::anomaly::AnomalyMode;


// work-load
Real64 BenchmarkHotgym::run(UInt EPOCHS, bool useSPlocal, bool useSPglobal, bool useTM, const UInt COLS, const UInt DIM_INPUT, const UInt CELLS) {
#ifndef NDEBUG
  EPOCHS = 2; // make test faster in Debug
#endif

  if(useTM ) {
	  NTA_CHECK(useSPlocal or useSPglobal) << "using TM requires a SP too";
  }

  std::cout << "starting test. DIM_INPUT=" << DIM_INPUT
  		<< ", DIM=" << COLS << ", CELLS=" << CELLS << std::endl;
  std::cout << "EPOCHS = " << EPOCHS << std::endl;


  // initialize SP, TM, Anomaly, AnomalyLikelihood
  tInit.start();
  ScalarEncoderParameters encParams;
  encParams.activeBits = 133;
  encParams.minimum = -100.0;
  encParams.maximum = 100.0;
  encParams.size = DIM_INPUT;
  ScalarEncoder enc( encParams );
  NTA_INFO << "SP (l) local inhibition is slow, so we reduce its data 10x smaller"; //to make it reasonably fast for test, for comparison x10
  SpatialPooler spGlobal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler spLocal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS/10u}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);

  TM tm(vector<UInt>{COLS}, CELLS);

  Anomaly an(5, AnomalyMode::PURE);
  Anomaly anLikelihood(5, AnomalyMode::LIKELIHOOD);
  tInit.stop();

  // data for processing input
  vector<UInt> input(DIM_INPUT);
  SDR inputSDR({DIM_INPUT});
  vector<UInt> outSP(COLS); // active array, output of SP/TM
  vector<UInt> outSPsparse;
  vector<UInt> outTM(COLS); 
  Real res = 0.0; //for anomaly:
  vector<UInt> prevPred_(COLS);
  Random rnd;

  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  tAll.start();

  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    //Input
//    generate(input.begin(), input.end(), [&] () { return rnd.getUInt32(2); });
    tRng.start();
    const Real r = (Real)(rnd.getUInt32(100) - rnd.getUInt32(100)*rnd.getReal64()); //rnd from range -100..100
    tRng.stop();

    //Encode
    tEnc.start();
    enc.encode(r, inputSDR);
    tEnc.stop();
    for(auto i = 0u; i < inputSDR.size; ++i) {
      input[i] = (UInt) inputSDR.getDense()[i];
    }

    //SP (global x local) 
    if(useSPlocal) {
    tSPloc.start();
    fill(outSP.begin(), outSP.end(), 0);
    spLocal.compute(input.data(), true, outSP.data());
    tSPloc.stop();
    NTA_CHECK(outSP.size() == COLS);
    }

    if(useSPglobal) {
    tSPglob.start();
    fill(outSP.begin(), outSP.end(), 0);
    spGlobal.compute(input.data(), true, outSP.data());
    tSPglob.stop();
    NTA_CHECK(outSP.size() == COLS);
    }
    outSPsparse = VectorHelpers::binaryToSparse(outSP);
    NTA_CHECK(outSPsparse.size() < COLS);


    // TM
    if(useTM) {
    tTM.start();
    tm.compute(outSPsparse.size(), outSPsparse.data(), true /*learn*/);
    const auto tmAct = tm.getActiveCells();
    tm.activateDendrites(); //must be called before getPredictiveCells 
    const auto tmPred = tm.getPredictiveCells();
    //TODO assert tmAct == spOut
    //TODO merge Act + Pred and use for anomaly from TM
    //TODO for anomaly: figure 1) use cols x cells? 2) use pred x { pred union active} ?
    //outTM = ...
    tTM.stop();
    }
 

    //Anomaly (pure x likelihood)
    tAn.start();
    res = an.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAn.stop();

    tAnLikelihood.start();
    anLikelihood.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAnLikelihood.stop();

    prevPred_ = outTM; //to be used as predicted T-1 //FIXME tmPred, or tmPred+Act?, also, cells->cols

    // print
    if (e == EPOCHS - 1) {
      tAll.stop();

      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << res << endl;
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(outSP), ",", "SP= ");
      NTA_CHECK(outSP[69] == 0) << "A value in SP computed incorrectly";
      cout << "==============TIMERS============" << endl;
      cout << "Init:\t" << tInit.getElapsed() << endl;
      cout << "Random:\t" << tRng.getElapsed() << endl;
      cout << "Encode:\t" << tEnc.getElapsed() << endl;
      if(useSPlocal)  cout << "SP (l):\t" << tSPloc.getElapsed() << "(x10)" << endl;
      if(useSPglobal) cout << "SP (g):\t" << tSPglob.getElapsed() << endl;
      if(useTM) cout << "TM:\t" << tTM.getElapsed() << endl;
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
  return tAll.getElapsed(); 
} //end run()
} //-ns
