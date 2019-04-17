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
#include "nupic/utils/Random.hpp"

namespace examples {

using namespace std;
using namespace nupic;
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
  encParams.sparsity = 0.3; //30% of the encoding are active bits (1's)
  encParams.minimum = -1.0;
  encParams.maximum = 1.0;
  encParams.size = DIM_INPUT; //the encoder is not optimal, it's to stress-test the SP,TM
  ScalarEncoder enc( encParams );
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
      17, 45, 73, 114, 141, 169, 183, 227, 244, 246, 258, 270, 308, 362, 423, 442, 478, 480, 509, 608, 675, 704, 719, 720, 726, 741, 764, 831, 844, 908, 941, 964, 971, 975, 1010, 1029, 1030, 1040, 1061, 1076, 1088, 1100, 1104, 1115, 1117, 1148, 1212, 1240, 1271, 1301, 1328, 1332, 1338, 1351, 1358, 1362, 1365, 1366, 1373, 1380, 1395, 1412, 1421, 1433, 1448, 1452, 1466, 1471, 1486, 1512, 1558, 1584, 1611, 1652, 1679, 1682, 1688, 1693, 1700, 1734, 1742, 1752, 1787, 1802, 1808, 1838, 1877, 1895, 1926, 1980, 1997, 2016, 2021, 2026, 2034, 2075, 2102, 2112, 2124, 2139, 2175, 2185, 2194, 2234, 2254, 2271, 2295, 2309, 2314, 2334, 2345, 2348, 2380, 2381, 2389, 2406, 2423, 2434, 2440, 2450, 2485, 2520, 2533, 2580, 2590, 2622, 2649, 2658, 2723, 2771, 2772, 2775, 2871, 2889, 2914, 2930, 2968, 2973, 2985, 2989, 2998, 3017, 3020, 3029, 3040, 3084, 3091, 3095, 3096, 3123, 3126, 3130, 3140, 3195, 3235, 3255, 3269, 3376, 3404, 3405, 3422, 3425, 3465, 3468, 3475, 3481, 3491, 3530, 3548, 3552, 3596, 3601, 3615, 3634, 3669, 3681, 3740, 3751, 3771, 3773, 3796, 3815, 3866, 3911, 3913, 3937, 3947, 3948, 4002, 4031, 4040, 4044, 4047, 4060, 4061, 4085, 4091, 4141, 4147, 4175, 4181, 4188, 4190, 4193, 4196, 4207, 4226, 4227, 4236, 4238, 4269, 4281, 4306, 4315, 4321, 4343, 4413, 4431, 4434, 4522, 4525, 4538, 4612, 4645, 4652, 4662, 4665, 4679, 4680, 4692, 4693, 4706, 4749, 4787, 4830, 4869, 4871, 4893, 4906, 4921, 4939, 4941, 4958, 4961, 4967, 4969, 4986, 4999, 5017, 5032, 5042, 5084, 5085, 5094, 5153, 5158, 5180, 5217, 5234, 5243, 5248, 5252, 5258, 5283, 5285, 5336, 5348, 5360, 5362, 5386, 5409, 5410, 5424, 5485, 5492, 5499, 5548, 5562, 5569, 5613, 5635, 5683, 5729, 5736, 5760, 5816, 5832, 5848, 5851, 5878, 5945, 5958, 6015, 6026, 6028, 6091, 6096, 6104, 6111, 6142, 6157, 6182, 6190, 6197, 6200, 6204, 6206, 6207, 6227, 6253, 6261, 6334, 6341, 6350, 6365, 6380, 6417, 6431, 6475, 6478, 6479, 6512, 6542, 6571, 6577, 6604, 6618, 6620, 6626, 6635, 6672, 6746, 6768, 6788, 6800, 6835, 6860, 6862, 6864, 6865, 6866, 6873, 6879, 6888, 6900, 6916, 6922, 6928, 6935, 6938, 6961, 7022, 7026, 7033, 7035, 7039, 7051, 7056, 7094, 7115, 7139, 7155, 7211, 7233, 7249, 7261, 7280, 7286, 7292, 7295, 7300, 7341, 7352, 7366, 7370, 7382, 7388, 7417, 7445, 7447, 7525, 7533, 7542, 7599, 7600, 7621, 7663, 7665, 7682, 7715, 7726, 7743, 7785, 7808, 7821, 7827, 7849, 7868, 7897, 7919, 7946, 7972, 7978, 7983, 7985, 7989, 7991, 7999, 8033, 8042, 8047, 8085, 8135, 8170, 8222, 8226, 8264, 8270, 8283, 8341, 8379, 8384, 8387, 8447, 8458, 8492, 8494, 8508, 8577, 8629, 8639, 8641, 8706, 8770, 8790, 8791, 8856, 8916, 8929, 8932, 8940, 8957, 8960, 8963, 8985, 9019, 9026, 9041, 9049, 9051, 9078, 9099, 9108, 9137, 9138, 9166, 9179, 9185, 9190, 9213, 9217, 9223, 9247, 9257, 9270, 9290, 9304, 9350, 9400, 9401, 9419, 9440, 9446, 9462, 9469, 9473, 9482, 9505, 9533, 9547, 9575, 9589, 9605, 9613, 9623, 9633, 9677, 9745, 9747, 9779, 9782, 9821, 9828, 9869, 9899, 9961, 9964, 9970, 9997, 9999
      };
      goldEnc.setSparse(deterministicEnc);

      SDR goldSP({COLS});
      const SDR_sparse_t deterministicSP{
      224, 277, 829, 906, 961, 1014, 1074, 1095, 1268, 1633
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
