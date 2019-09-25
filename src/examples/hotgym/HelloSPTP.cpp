/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2015, Numenta, Inc.
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

#include <algorithm> // std::generate
#include <iostream>
#include <vector>

#include "HelloSPTP.hpp"

#include "htm/algorithms/TemporalMemory.hpp"
#include "htm/algorithms/SpatialPooler.hpp"
#include "htm/encoders/RandomDistributedScalarEncoder.hpp"
#include "htm/algorithms/AnomalyLikelihood.hpp"

#include "htm/types/Sdr.hpp"
#include "htm/utils/Random.hpp"
#include "htm/utils/MovingAverage.hpp"
#include "htm/utils/SdrMetrics.hpp"

namespace examples {

using namespace std;
using namespace htm;


// work-load
Real64 BenchmarkHotgym::run(UInt EPOCHS, bool useSPlocal, bool useSPglobal, bool useTM, const UInt COLS, const UInt DIM_INPUT, const UInt CELLS)
{
#ifndef NDEBUG
EPOCHS = 2; // make test faster in Debug
#endif

#if defined __aarch64__ || defined __arm__
#undef _ARCH_DETERMINISTIC
#else
#define _ARCH_DETERMINISTIC
#endif

  if(useTM ) {
	  NTA_CHECK(useSPlocal or useSPglobal) << "using TM requires a SP too";
  }

  std::cout << "starting test. DIM_INPUT=" << DIM_INPUT
  		<< ", DIM=" << COLS << ", CELLS=" << CELLS << std::endl;
  std::cout << "EPOCHS = " << EPOCHS << std::endl;


  // initialize SP, TM, AnomalyLikelihood
  tInit.start();
  RDSE_Parameters encParams;
  encParams.sparsity = 0.2f; //20% of the encoding are active bits (1's)
  encParams.size = DIM_INPUT; //the encoder is not optimal, it's to stress-test the SP,TM
//  encParams.resolution = 0.002f;
  encParams.radius = 0.03f;
  encParams.seed = 2019u;
  RandomDistributedScalarEncoder enc( encParams );
  SpatialPooler spGlobal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with globalInh
  SpatialPooler  spLocal(enc.dimensions, vector<UInt>{COLS}); // Spatial pooler with local inh
  spGlobal.setGlobalInhibition(true);
  spLocal.setGlobalInhibition(false);
  Random rnd(42); //uses fixed seed for deterministic output checks

  TemporalMemory tm(vector<UInt>{COLS}, CELLS);

  AnomalyLikelihood anLikelihood;
  tInit.stop();

  // data for processing input
  SDR input(enc.dimensions);
  SDR outSPglobal(spGlobal.getColumnDimensions()); // active array, output of SP/TM
  SDR outSPlocal(spLocal.getColumnDimensions()); //for SPlocal
  SDR outSP(vector<UInt>{COLS});
  SDR outTM(spGlobal.getColumnDimensions()); 
  Real an = 0.0f, anLikely = 0.0f; //for anomaly:
  MovingAverage avgAnom10(1000); //chose the window large enough so there's (some) periodicity in the patter, so TM can learn something

  //metrics
  Metrics statsInput(input, 1000);
  Metrics statsSPlocal(outSPlocal, 1000);
  Metrics statsSPglobal(outSPglobal, 1000);
  Metrics statsTM(outTM, 1000);

  /*
   * For example: fn = sin(x) -> periodic >= 2Pi ~ 6.3 && x+=0.01 -> 630 steps to 1st period -> window >= 630
   */
  Real avgAnomOld_ = 1.0;
  NTA_CHECK(avgAnomOld_ >= avgAnom10.getCurrentAvg()) << "TM should learn and avg anomalies improve, but we got: "
    << avgAnomOld_ << " and now: " << avgAnom10.getCurrentAvg(); //invariant


  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  tAll.start();

  //run
  float x=0.0f;
  for (UInt e = 0; e < EPOCHS; e++) { //FIXME EPOCHS is actually steps, there's just 1 pass through data/epoch.

    //Encode
    tEnc.start();
    x+=0.01f; //step size for fn(x)
    enc.encode(sin(x), input); //model sin(x) function //TODO replace with CSV data
//    cout << x << "\n" << sin(x) << "\n" << input << "\n\n";
    tEnc.stop();

    tRng.start();
    input.addNoise(0.01f, rnd); //change 1% of the SDR for each iteration, this makes a random sequence, but seemingly stable
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
      outTM = tm.cellsToColumns( tm.getPredictiveCells() );
      tTM.stop();
    }


    //Anomaly (pure x likelihood)
    an = tm.anomaly;
    avgAnom10.compute(an); //moving average
    if(e % 1000 == 0) {
      NTA_CHECK(avgAnomOld_ >= avgAnom10.getCurrentAvg()) << "TM should learn and avg anomalies improve, but we got: "
        << avgAnomOld_ << " and now: " << avgAnom10.getCurrentAvg(); //invariant
      avgAnomOld_ = avgAnom10.getCurrentAvg(); //update
    }
    tAnLikelihood.start();
    anLikelihood.anomalyProbability(an); //FIXME AnLikelihood is 0.0, probably not working correctly
    tAnLikelihood.stop();


    // print
    if (e == EPOCHS - 1) {
      tAll.stop();

      //print connections stats
      cout << "\nInput :\n" << statsInput
	   << "\nSP(local) " << spLocal.connections
	   << "\nSP(local) " << statsSPlocal
           << "\nSP(global) " << spGlobal.connections
	   << "\nSP(global) " << statsSPglobal
           << "\nTM " << tm.connections 
	   << "\nTM " << statsTM
	   << "\n";

      // output values
      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << an << endl;
      cout << "Anomaly (avg) = " << avgAnom10.getCurrentAvg() << endl;
      cout << "Anomaly (Likelihood) = " << anLikely << endl;
      cout << "SP (g)= " << outSP << endl;
      cout << "SP (l)= " << outSPlocal <<endl;
      cout << "TM= " << outTM << endl;

      //timers
      cout << "==============TIMERS============" << endl;
      cout << "Init:\t" << tInit.getElapsed() << endl;
      cout << "Random:\t" << tRng.getElapsed() << endl;
      cout << "Encode:\t" << tEnc.getElapsed() << endl;
      if(useSPlocal)  cout << "SP (l):\t" << tSPloc.getElapsed()*1.0f  << endl;
      if(useSPglobal) cout << "SP (g):\t" << tSPglob.getElapsed() << endl;
      if(useTM) cout << "TM:\t" << tTM.getElapsed() << endl;
      cout << "AN:\t" << tAnLikelihood.getElapsed() << endl;

      // check deterministic SP, TM output 
      SDR goldEnc({DIM_INPUT});
      const SDR_sparse_t deterministicEnc{
        0, 4, 13, 21, 24, 30, 32, 37, 40, 46, 47, 48, 50, 51, 64, 68, 79, 81, 89, 97, 99, 114, 120, 135, 136, 140, 141, 143, 144, 147, 151, 155, 161, 162, 164, 165, 169, 172, 174, 179, 181, 192, 201, 204, 205, 210, 213, 226, 227, 237, 242, 247, 249, 254, 255, 262, 268, 271, 282, 283, 295, 302, 306, 307, 317, 330, 349, 353, 366, 380, 383, 393, 404, 409, 410, 420, 422, 441,446, 447, 456, 458, 464, 468, 476, 497, 499, 512, 521, 528, 531, 534, 538, 539, 541, 545, 550, 557, 562, 565, 575, 581, 589, 592, 599, 613, 617, 622, 647, 652, 686, 687, 691, 699, 704, 710, 713, 716, 722, 729, 736, 740, 747, 749, 753, 754, 758, 766, 778, 790, 791, 797, 800, 808, 809, 812, 815, 826, 828, 830, 837, 852, 853, 856, 863, 864, 873, 878, 882, 885, 893, 894, 895, 905, 906, 914, 915, 920, 924, 927, 937, 939, 944, 947, 951, 954, 956, 967, 968, 969, 973, 975, 976, 979, 981, 991, 998
      };
      goldEnc.setSparse(deterministicEnc);

      SDR goldSP({COLS});
      const SDR_sparse_t deterministicSP{
        62, 72, 73, 82, 85, 102, 263, 277, 287, 303, 306, 308, 309, 322, 337, 339, 340, 352, 370, 493, 1094, 1095, 1114, 1115, 1120, 1463, 1512, 1518, 1647, 1651, 1691, 1694, 1729, 1745, 1746, 1760, 1770, 1774, 1775, 1781, 1797, 1798, 1803, 1804, 1805, 1812, 1827, 1828, 1831, 1832, 1858, 1859, 1860, 1861, 1862, 1875, 1878, 1880, 1881, 1898, 1918, 1923, 1929, 1931,1936, 1950, 1953, 1956, 1958, 1961, 1964, 1965, 1967, 1971, 1973, 1975, 1976, 1979, 1980, 1981, 1982, 1984, 1985, 1986, 1988, 1991, 1994, 1996, 1997, 1998, 1999, 2002, 2006, 2008, 2011, 2012, 2013, 2017, 2019, 2022, 2027, 2030
      };
      goldSP.setSparse(deterministicSP);

      SDR goldSPlocal({COLS});
      const SDR_sparse_t deterministicSPlocal{
        12, 13, 71, 72, 75, 78, 82, 85, 131, 171, 182, 186, 189, 194, 201, 263, 277, 287, 308, 319, 323, 337, 339, 365, 407, 423, 429, 432, 434, 445, 493, 494, 502, 508, 523, 542, 554, 559, 585, 586, 610, 611, 612, 644, 645, 647, 691, 698, 699, 701, 702, 707, 777, 809, 810, 811, 833, 839, 841, 920, 923, 928, 929, 935, 955, 1003, 1005, 1073, 1076, 1094, 1095, 1114,1115, 1133, 1134, 1184, 1203, 1214, 1232, 1233, 1244, 1253, 1268, 1278, 1291, 1294, 1306, 1309, 1331, 1402, 1410, 1427, 1434, 1442, 1463, 1508, 1512, 1514, 1515, 1518, 1561, 1564, 1590, 1623, 1626, 1630, 1647, 1651, 1691, 1694, 1729, 1745, 1746, 1760, 1797, 1804, 1805, 1812, 1827, 1858, 1860, 1861, 1862, 1918, 1956, 1961, 1965, 1971, 1975, 1994, 2012
      };
      goldSPlocal.setSparse(deterministicSPlocal);

      SDR goldTM({COLS});
      const SDR_sparse_t deterministicTM{
        62, 77, 85, 322, 340, 432, 952, 1120, 1488, 1502, 1512, 1518, 1547, 1627, 1633, 1668, 1727, 1729, 1797, 1803, 1805, 1812, 1858, 1859, 1896, 1918, 1923, 1925, 1929, 1931, 1939, 1941, 1942, 1944, 1950, 1953, 1955, 1956, 1965, 1966, 1967, 1968, 1974, 1980, 1987, 1996, 2006, 2008, 2011, 2027, 2030, 2042, 2046
      };
      goldTM.setSparse(deterministicTM);

      const float goldAn    = 0.627451f;
      const float goldAnAvg = 0.407265f;

#ifdef _ARCH_DETERMINISTIC
      if(EPOCHS == 5000) {
        //these hand-written values are only valid for EPOCHS = 5000 (default), but not for debug and custom runs.
        NTA_CHECK(input == goldEnc) << "Deterministic output of Encoder failed!\n" << input << "should be:\n" << goldEnc;
        if(useSPglobal) { NTA_CHECK(outSPglobal == goldSP) << "Deterministic output of SP (g) failed!\n" << outSP << "should be:\n" << goldSP; }
        if(useSPlocal) {  NTA_CHECK(outSPlocal == goldSPlocal) << "Deterministic output of SP (l) failed!\n" << outSPlocal << "should be:\n" << goldSPlocal; }
        if(useTM) {       NTA_CHECK(outTM == goldTM) << "Deterministic output of TM failed!\n" << outTM << "should be:\n" << goldTM; }
        NTA_CHECK(static_cast<UInt>(an *10000.0f) == static_cast<UInt>(goldAn *10000.0f)) //compare to 4 decimal places
                  << "Deterministic output of Anomaly failed! " << an << "should be: " << goldAn;
        NTA_CHECK(static_cast<UInt>(avgAnom10.getCurrentAvg() * 10000.0f) == static_cast<UInt>(goldAnAvg * 10000.0f))
                  << "Deterministic average anom score failed:" << avgAnom10.getCurrentAvg() << " should be: " << goldAnAvg;
      }
#endif

      // check runtime speed
      const size_t timeTotal = (size_t)floor(tAll.getElapsed());
      cout << "Total elapsed time = " << timeTotal << " seconds" << endl;
      if(EPOCHS >= 100) { //show only relevant values, ie don't run in valgrind (ndebug, epochs=5) run
#ifdef NTA_OS_LINUX
        const size_t CI_avg_time = (size_t)floor(99*Timer::getSpeed()); //sec //FIXME the CI speed broken for docker linux
        NTA_CHECK(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
          "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time << "), speed coef.= " << Timer::getSpeed();
#endif
      }
    }
  } //end for
  return tAll.getElapsed();
} //end run()
} //-ns
