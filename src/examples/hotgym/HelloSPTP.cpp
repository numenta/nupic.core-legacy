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

#include "nupic/algorithms/TemporalMemory.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/encoders/RandomDistributedScalarEncoder.hpp"
#include "nupic/algorithms/AnomalyLikelihood.hpp"

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
using nupic::algorithms::anomaly::AnomalyLikelihood;


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


  // initialize SP, TM, AnomalyLikelihood
  tInit.start();
  EncoderParameters encParams;
  encParams.sparsity = 0.2f; //20% of the encoding are active bits (1's)
  encParams.size = DIM_INPUT; //the encoder is not optimal, it's to stress-test the SP,TM
//  encParams.resolution = 0.002f;
  encParams.radius = 0.03f;
  encParams.seed = 2019u;
  Encoder enc( encParams );
  SpatialPooler spGlobal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler  spLocal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);
  Random rnd(1); //uses fixed seed for deterministic output checks

  TM tm(vector<UInt>{COLS}, CELLS);

  AnomalyLikelihood anLikelihood;
  tInit.stop();

  // data for processing input
  SDR input(enc.dimensions);
  SDR outSPglobal(spGlobal.getColumnDimensions()); // active array, output of SP/TM
  SDR outSPlocal(spLocal.getColumnDimensions()); //for SPlocal
  SDR outSP(vector<UInt>{COLS});
  SDR outTM(spGlobal.getColumnDimensions()); 
  Real anLikely = 0.0f; //for anomaly:

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
//    cout << x << "\n" << sin(x) << "\n" << input << "\n\n";
    tEnc.stop();

    tRng.start();
    input.addNoise(0.01, rnd); //change 1% of the SDR for each iteration, this makes a random sequence, but seemingly stable
    tRng.stop();

    //SP (global x local)
    if(useSPlocal) {
    tSPloc.start();
    spLocal.compute(input, true, outSPlocal);
    tSPloc.stop();
    }

    if(useSPglobal) {
    tSPglob.start();
    spGlobal.compute(input, true, outSPglobal);
    tSPglob.stop();
    }
    outSP = outSPglobal; //toggle if local/global SP is used further down the chain (TM, Anomaly)

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
    tAnLikelihood.start();
    anLikelihood.anomalyProbability(tm.anomaly.score); //FIXME AnLikelihood is 0.0, probably not working correctly
    tAnLikelihood.stop();


    // print
    if (e == EPOCHS - 1) {
      tAll.stop();

      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << tm.anomaly.score << endl;
      cout << "Anomaly (Likelihood) = " << anLikely << endl;
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
        0, 4, 13, 21, 24, 30, 32, 37, 40, 46, 47, 48, 50, 51, 64, 68, 79, 81, 89, 97, 99, 114, 120, 135, 136, 140, 141, 143, 144, 147, 151, 155, 161, 162, 164, 165, 169, 172, 174, 179, 181, 192, 201, 204, 205, 210, 213, 226, 237, 242, 247, 249, 254, 255, 262, 268, 271, 282, 283, 295, 302, 306, 307, 317, 330, 349, 353, 366, 368, 380, 393, 399, 404, 409, 410, 420, 422, 441,446, 447, 456, 458, 464, 468, 476, 497, 499, 512, 521, 528, 531, 534, 538, 539, 541, 545, 550, 557, 562, 565, 575, 581, 589, 592, 599, 613, 617, 622, 647, 652, 686, 687, 691, 699, 704, 710, 713, 716, 722, 729, 736, 740, 747, 749, 753, 754, 758, 766, 778, 790, 791, 797, 800, 808, 809, 812, 815, 826, 828, 830, 837, 838, 852, 853, 856, 863, 864, 873, 878, 882, 885, 893, 894, 895, 905, 906, 914, 915, 920, 924, 927, 937, 939, 944, 947, 951, 954, 956, 967, 968, 969, 973, 975, 976, 981, 991, 998
      };
      goldEnc.setSparse(deterministicEnc);

      SDR goldSP({COLS});
      const SDR_sparse_t deterministicSP{
        72, 75, 284, 303, 305, 317, 329, 525, 1095, 2027
      };
      goldSP.setSparse(deterministicSP);

      SDR goldSPlocal({COLS});
      const SDR_sparse_t deterministicSPlocal{
        6, 12, 26, 57, 63, 72, 75, 76, 77, 80, 82, 103, 105, 124, 135, 154, 171, 174, 175, 185, 192, 193, 195, 198, 263, 284, 296, 302, 303, 305, 313, 317, 319, 320, 356, 363, 364, 401, 403, 404, 410, 413, 425, 426, 428, 449, 491, 496, 511, 515, 516, 518, 520, 525, 529, 536, 550, 556, 574, 583, 592, 597, 598, 603, 609, 622, 626, 636, 645, 652, 704, 706, 722, 726, 727, 728, 729, 747, 751, 766, 779, 808, 833, 837, 838, 840, 848, 850, 853, 860, 908, 912, 918, 919, 923, 927, 929, 930, 931, 932, 970, 989, 1006, 1038, 1066, 1082, 1085, 1087, 1092, 1094, 1095, 1113, 1115, 1125, 1128, 1174, 1179, 1180, 1182, 1185, 1205, 1206, 1232, 1236, 1238, 1239, 1240, 1245, 1271, 1292, 1295, 1300, 1303, 1307, 1311, 1319, 1320, 1322, 1382, 1401, 1412, 1415, 1421, 1426, 1431, 1434, 1438, 1470, 1474, 1492, 1501, 1511, 1521, 1524, 1525, 1530, 1532, 1537, 1540, 1600, 1617, 1620, 1622, 1632, 1638, 1641, 1667, 1672, 1680, 1684, 1686, 1690, 1699, 1702, 1742, 1744, 1745, 1746, 1765, 1770, 1774, 1801, 1807, 1808, 1816, 1830, 1834, 1849, 1861, 1867, 1871, 1882, 1902, 1907, 1943, 1945, 1955, 1956, 1966, 1968, 1969, 1971, 1986, 2018, 2025, 2027
      };
      goldSPlocal.setSparse(deterministicSPlocal);

      SDR goldTM({COLS});
      const SDR_sparse_t deterministicTM{
        26, 75 
      };
      goldTM.setSparse(deterministicTM);

      const float goldAn = 0.8f;

      if(EPOCHS == 5000) { //these hand-written values are only valid for EPOCHS = 5000 (default), but not for debug and custom runs. 
        NTA_CHECK(input == goldEnc) << "Deterministic output of Encoder failed!\n" << input << "should be:\n" << goldEnc;
        NTA_CHECK(outSPglobal == goldSP) << "Deterministic output of SP (g) failed!\n" << outSP << "should be:\n" << goldSP;
	NTA_CHECK(outSPlocal == goldSPlocal) << "Deterministic output of SP (l) failed!\n" << outSPlocal << "should be:\n" << goldSPlocal;
#ifndef _MSC_VER //FIXME deterministic checks fail on Windows
        NTA_CHECK(outTM == goldTM) << "Deterministic output of TM failed!\n" << outTM << "should be:\n" << goldTM; 
        NTA_CHECK(static_cast<UInt>(tm.anomaly.score *10000) == static_cast<UInt>(goldAn *10000)) //compare to 4 decimal places
		<< "Deterministic output of Anomaly failed! " << tm.anomaly.score << "should be: " << goldAn;
#endif
      }

      // check runtime speed
      const size_t timeTotal = (size_t)floor(tAll.getElapsed());
      cout << "Total elapsed time = " << timeTotal << " seconds" << endl;
      if(EPOCHS >= 100) { //show only relevant values, ie don't run in valgrind (ndebug, epochs=5) run
#ifndef _MSC_VER
        const size_t CI_avg_time = (size_t)floor(30*Timer::getSpeed()); //sec
        NTA_CHECK(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
          "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time << "), speed coef.= " << Timer::getSpeed();
#endif
      }
    }
  } //end for
  return tAll.getElapsed();
} //end run()
} //-ns
