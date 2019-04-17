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
#include "nupic/encoders/ScalarEncoder.hpp"

#include "nupic/types/Sdr.hpp"
#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/Random.hpp"

namespace examples {

using namespace std;
using namespace nupic;
using namespace nupic::utils;
using namespace nupic::sdr;

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
  Random rnd(1); //FIXME fix bug in Random, where static_gen must not be used! change this '1' and all breaks
  NTA_INFO << "SP (l) local inhibition is slow, so we reduce its data 10x smaller"; //to make it reasonably fast for test, for comparison x10
  SpatialPooler spGlobal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler spLocal(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);

  TM tm(vector<UInt>{COLS}, CELLS);

  Anomaly an(5, AnomalyMode::PURE);
  Anomaly anLikelihood(5, AnomalyMode::LIKELIHOOD);
  tInit.stop();

  // data for processing input
  SDR input({DIM_INPUT});
  SDR outSP({COLS}); // active array, output of SP/TM
  SDR outSPlocal({COLS}); //for SPlocal
  SDR outTM({COLS}); 
  Real res = 0.0; //for anomaly:
  SDR prevPred_({COLS}); //holds T-1 TM.predictive cells

  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  tAll.start();

  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    tRng.start();
    const Real r = rnd.realRange(-100.0f, 100.0f);
    tRng.stop();

    //Encode
    tEnc.start();
    enc.encode(r, input);
    tEnc.stop();

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
    tm.activateDendrites(); //must be called before getPredictiveCells
    SDR cells({CELLS*COLS});
    cells.setSparse(tm.getPredictiveCells());
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
      const SDR_sparse_t deterministicEnc{5393, 5394, 5395, 5396, 5397, 5398, 5399, 5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 5409, 5410, 5411, 5412, 5413, 5414, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449, 5450, 5451, 5452, 5453,5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5463, 5464, 5465, 5466, 5467, 5468, 5469, 5470, 5471, 5472, 5473, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516,5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525};
      goldEnc.setSparse(deterministicEnc);

      SDR goldSP({COLS});
      const SDR_sparse_t deterministicSP{1110, 1115, 1116, 1119, 1121, 1122, 1123, 1124, 1125, 1128};
      goldSP.setSparse(deterministicSP);

      SDR goldSPlocal({COLS});
      const SDR_sparse_t deterministicSPlocal{};
      goldSPlocal.setSparse(deterministicSPlocal);

      SDR goldTM({COLS});
      const SDR_sparse_t deterministicTM{}; //TODO use hotgym CSV data, not random. Otherwise TM does not learn, and this field is empty
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
