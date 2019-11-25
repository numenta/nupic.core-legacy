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

#include <htm/engine/Network.hpp>
#include <htm/utils/Random.hpp>
#include <htm/utils/Log.hpp>
#include <htm/ntypes/Value.hpp>
#include "htm/utils/MovingAverage.hpp"
#include "htm/algorithms/AnomalyLikelihood.hpp"
#include <fstream>

using namespace htm;

static bool verbose = true;
#define VERBOSE if (verbose)  std::cout << "    "


//this runs as an executable

int main(int argc, char* argv[]) {
  htm::UInt EPOCHS = 5000;      // number of iterations (calls to encoder/SP/TP compute() )
#ifndef NDEBUG
  EPOCHS = 2; // make test faster in Debug
#endif

  const UInt DIM_INPUT = 1000;  // Width of encoder output
  const UInt COLS = 2048;       // number of columns in SP, TP
  const UInt CELLS = 8;         // cells per column in TP
  Random rnd(42);               // uses fixed seed for deterministic output
  std::ofstream ofs;

  std::string encoder_params   = "{size: " + std::to_string(DIM_INPUT) + ", sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}";
  std::string sp_global_params = "{columnCount: " + std::to_string(COLS) + ", globalInhibition: true}";
  std::string tm_params        = "{cellsPerColumn: " + std::to_string(CELLS) + ", orColumnOutputs: true}";

  //  Runtime arguments:  napi_sine [epochs [filename]]
  if(argc >= 2) {
    EPOCHS = std::stoi(argv[1]);         // number of iterations (default 5000)
  }
  if (argc >= 3) {
    ofs.open(argv[2], std::ios::out);    // output filename (for plotting)
  }

  try {

    std::cout << "initializing. DIM_INPUT=" << DIM_INPUT << ", COLS=" << COLS << ", CELLS=" << CELLS << "\n";

    Network net;

    // Declare the regions to use
    std::shared_ptr<Region> encoder   = net.addRegion("encoder",   "RDSERegion", encoder_params);
    std::shared_ptr<Region> sp_global = net.addRegion("sp_global", "SPRegion",   sp_global_params);
    std::shared_ptr<Region> tm        = net.addRegion("tm",        "TMRegion",   tm_params);

    // Setup data flows between regions
    net.link("encoder",   "sp_global", "", "", "encoded", "bottomUpIn");
    net.link("sp_global", "tm",        "", "", "bottomUpOut", "bottomUpIn");

    net.initialize();


    ///////////////////////////////////////////////////////////////
    //
    //                          .----------------.
    //                         |    encoder      |
    //                 data--->|  (RDSERegion)   |
    //                         |                 |
    //                         `-----------------'
    //                                 |
    //                         .-----------------.
    //                         |   sp_global     |
    //                         |  (SPRegion)     |
    //                         |                 |
    //                         `-----------------'
    //                                 |
    //                         .-----------------.
    //                         |      tm         |
    //                         |   (TMRegion)    |
    //                         |                 |
    //                         `-----------------'
    //
    //////////////////////////////////////////////////////////////////


    // enable this to see a trace as it executes
    //net.setLogLevel(LogLevel::LogLevel_Verbose);

    std::cout << "Running: " << EPOCHS << " Iterations.\n ";

    float anLikely = 0.0f;
    MovingAverage avgAnomaly(1000); 
    AnomalyLikelihood anLikelihood;

    // RUN
    float x = 0.00f; 
    for (size_t e = 0; e < EPOCHS; e++) {
      // genarate some data to send to the encoder
      
      //  -- A sine wave, one degree rotation per iteration (an alternate function)
      //double data = std::sin(i * (3.1415 / 180));
      
      // -- sine wave, 0.01 radians per iteration   (Note: first iteration is for x=0.01, not 0)
      x += 0.01f; // step size for fn(x)
      double data = std::sin(x);
      encoder->setParameterReal64("sensedValue", data); // feed data into RDSE encoder for this iteration.

      // Execute an iteration.
      net.run(1);

      float an = ((float *)tm->getOutputData("anomaly").getBuffer())[0];
      avgAnomaly.compute(an);
      anLikely = anLikelihood.anomalyProbability(an); 


      // Save the data for plotting.   <iteration>, <sin data>, <anomaly>, <likelyhood>\n
      if (ofs.is_open()) {
        ofs << e << "," << data << "," << an << "," << anLikely << std::endl;
      }

      if (e == EPOCHS - 1) 
      {

        // output values
        float final_an = ((float *)tm->getOutputData("anomaly").getBuffer())[0];
        VERBOSE << "Result after " << e + 1 << " iterations.\n";
        VERBOSE << "  Anomaly(avg)        = " << avgAnomaly.getCurrentAvg() << std::endl;
        VERBOSE << "  Anomaly(Likelihood) = " << anLikely << endl;
        VERBOSE << "  Encoder out         = " << encoder->getOutputData("encoded").getSDR();
        VERBOSE << "  SP (global)         = " << sp_global->getOutputData("bottomUpOut").getSDR();
        VERBOSE << "  TM predictive       = " << tm->getOutputData("predictiveCells").getSDR();
      }
    }
    if (ofs.is_open())
      ofs.close();


    std::cout << "finished\n";


  } catch (Exception &ex) {
    std::cerr << ex.what();
    if (ofs.is_open())
      ofs.close();
    return 1;
  }
  
  return 0;
}

