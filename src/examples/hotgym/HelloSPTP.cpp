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

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/algorithms/TemporalMemory.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/encoders/RandomDistributedScalarEncoder.hpp"

#include "nupic/types/Sdr.hpp"
#include "nupic/utils/Random.hpp"

namespace examples {

using namespace std;
using namespace nupic;
using namespace nupic::sdr;

using Encoder = nupic::encoders::RandomDistributedScalarEncoder;
using EncoderParameters = nupic::encoders::RDSE_Parameters;
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
  EncoderParameters encParams;
  encParams.sparsity = 0.3f; //30% of the encoding are active bits (1's)
  encParams.size = DIM_INPUT; //the encoder is not optimal, it's to stress-test the SP,TM
  encParams.resolution = 0.002f;
  Encoder enc( encParams );
  SpatialPooler spGlobal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler  spLocal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);
  Random rnd(1);

  TM tm(spGlobal.getColumnDimensions(), CELLS);

  Anomaly an(5, AnomalyMode::PURE);
  Anomaly anLikelihood(5, AnomalyMode::LIKELIHOOD);
  tInit.stop();

  // data for processing input
  SDR input(enc.dimensions);
  SDR outSP(spGlobal.getColumnDimensions()); // active array, output of SP/TM
  SDR outSPlocal(spLocal.getColumnDimensions()); //for SPlocal
  SDR outTM(spGlobal.getColumnDimensions()); 
  Real res = 0.0; //for anomaly:
  SDR prevPred_(outTM.dimensions); //holds T-1 TM.predictive cells

  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  tAll.start();

  //run
  float x=0.0f;
  for (UInt e = 0; e < EPOCHS; e++) {

    //Encode
    tEnc.start();
    x+=0.01f; //step size for fn(x)
    enc.encode(sin(x), input); //model sin(x) function //TODO replace with CSV data
    tEnc.stop();

    tRng.start();
    input.randomize(0.01, rnd); //change 1% of the SDR for each iteration, this makes a random sequence, but seemingly stable
    tRng.stop();

    //SP (global x local)
    if(useSPlocal) {
    tSPloc.start();
    spLocal.compute(input, true, outSPlocal);
    tSPloc.stop();
    }

    if(useSPglobal) {
    tSPglob.start();
    spGlobal.compute(input, true, outSP);
    tSPglob.stop();
    }

    // TM
    if(useTM) {
    tTM.start();
    tm.compute(outSP, true /*learn*/); //to uses output of SPglobal
    tm.activateDendrites(); //required to enable tm.getPredictiveCells()
    SDR cells({CELLS*COLS});
    tm.getPredictiveCells(cells);
    outTM = tm.cellsToColumns(cells);
    tTM.stop();
    }


    //Anomaly (pure x likelihood)
    tAn.start();
    res = an.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAn.stop();

    tAnLikelihood.start();
    anLikelihood.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    tAnLikelihood.stop();

    prevPred_ = outTM; //to be used as predicted T-1

    // print
    if (e == EPOCHS - 1) {
      tAll.stop();

      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << res << endl;
      cout << "SP (g)= " << outSP << endl;
      cout << "SP (l)= " << outSPlocal <<endl;
      cout << "TM= " << outTM << endl;
      cout << "==============TIMERS============" << endl;
      cout << "Init:\t" << tInit.getElapsed() << endl;
      cout << "Random:\t" << tRng.getElapsed() << endl;
      cout << "Encode:\t" << tEnc.getElapsed() << endl;
      if(useSPlocal)  cout << "SP (l):\t" << tSPloc.getElapsed()*1.0f  << endl;
      if(useSPglobal) cout << "SP (g):\t" << tSPglob.getElapsed() << endl;
      if(useTM) cout << "TM:\t" << tTM.getElapsed() << endl;
      cout << "AN:\t" << tAn.getElapsed() << endl;
      cout << "AN:\t" << tAnLikelihood.getElapsed() << endl;

      // check deterministic SP, TM output 
      SDR goldEnc({DIM_INPUT});
      const SDR_sparse_t deterministicEnc{
        227, 246, 362, 704, 726, 1010, 1040, 1100, 1117, 1240, 1338, 1366, 1373, 1395, 1433, 1611, 1682, 1693, 1700, 1877, 1926, 2112, 2139, 2345, 2450, 2533, 2580, 2649, 2658, 2973, 2998, 3084, 3096, 3235, 3376, 3425, 3465, 3740, 4085, 4147, 4181, 4306, 4645, 4662, 4680, 4871, 4941, 4958, 4986, 4999, 5085, 5153, 5248, 5285, 5386, 5424, 5736, 5816, 5832, 5851, 5945, 6026, 6028, 6104, 6182, 6206, 6207, 6253, 6475, 6571, 6604, 6900, 6916, 6928, 6938, 7026, 7115, 7139, 7417, 7533, 7682, 7715, 7821, 7868, 7999, 8033, 8270, 8641, 8706, 9019, 9138, 9217, 9350, 9469, 9547, 9589, 9623, 9828, 9869, 9899
      };
      goldEnc.setSparse(deterministicEnc);

      SDR goldSP({COLS});
      const SDR_sparse_t deterministicSP{
        253, 277, 502, 542, 701, 1011, 1268, 1459, 1783, 1870
      };
      goldSP.setSparse(deterministicSP);

      SDR goldSPlocal({COLS});
      const SDR_sparse_t deterministicSPlocal{}; //FIXME SP local inh returns almost all active cols, probably incorrect settings
      goldSPlocal.setSparse(deterministicSPlocal);

      SDR goldTM({COLS});
      const SDR_sparse_t deterministicTM{
      
      }; //FIXME why TM does not learn, and this field is empty?!
      goldTM.setSparse(deterministicTM);

      const float goldAn = 1.0f;

      if(EPOCHS == 5000) { //these hand-written values are only valid for EPOCHS = 5000 (default), but not for debug and custom runs. 
        NTA_CHECK(input == goldEnc) << "Deterministic output of Encoder failed!\n" << input << "should be:\n" << goldEnc;
        NTA_CHECK(outSP == goldSP) << "Deterministic output of SP (g) failed!\n" << outSP << "should be:\n" << goldSP;
	//TODO enable the check: NTA_CHECK(outSPlocal == goldSPlocal) << "Deterministic output of SP (l) failed!\n" << outSPlocal << "should be:\n" << goldSPlocal;
        NTA_CHECK(outTM == goldTM) << "Deterministic output of TM failed!\n" << outTM << "should be:\n" << goldTM; 
        NTA_CHECK(res == goldAn) << "Deterministic output of Anomaly failed! " << res << "should be: " << goldAn;
      }

      // check runtime speed
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
