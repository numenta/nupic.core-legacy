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
#include <fstream>

using namespace htm;

static bool verbose = false;
#define VERBOSE if (verbose)  std::cout << "    "


//this runs as an executable

int main(int argc, char* argv[]) {
  htm::UInt EPOCHS = 5000;      // number of iterations (calls to encoder/SP/TP compute() )
  const UInt DIM_INPUT = 1000;  // Width of encoder output
  const UInt COLS = 2048;       // number of columns in SP, TP
  const UInt CELLS = 8;         // cells per column in TP
  Random rnd(42);               // uses fixed seed for deterministic output
  std::ofstream ofs;

  std::string encoder_params   = "{size: " + std::to_string(DIM_INPUT) + ", activeBits: 4, radius: 0.5, seed: 2019 }";
//  std::string sp_local_params  = "{columnCount: " + std::to_string(COLS) + "}";
  std::string sp_global_params = "{columnCount: " + std::to_string(COLS) + ", globalInhibition: true}";
  std::string tm_params        = "{activationThreshold: 9, cellsPerColumn: " + std::to_string(CELLS) + "}";

  //  Runtime arguments:  napi_sine [epochs [filename]]
  if(argc >= 2) {
    EPOCHS = std::stoi(argv[1]);         // number of iterations (default 500)
  }
  if (argc >= 3) {
    ofs.open(argv[2], std::ios::out);    // output filename (for plotting)
  }

  try {

    std::cout << "initializing\n";

    Network net;

    // Declare the regions to use
    std::shared_ptr<Region> encoder   = net.addRegion("encoder",   "RDSERegion", encoder_params);
//    std::shared_ptr<Region> sp_local  = net.addRegion("sp_local",  "SPRegion",   sp_local_params);
    std::shared_ptr<Region> sp_global = net.addRegion("sp_global", "SPRegion",   sp_global_params);
    std::shared_ptr<Region> tm        = net.addRegion("tm",        "TMRegion",   tm_params);

    // Setup data flows between regions
//    net.link("encoder",   "sp_local",  "", "", "encoded", "bottomUpIn");
    net.link("encoder",   "sp_global", "", "", "encoded", "bottomUpIn");
    net.link("sp_global", "tm",        "", "", "bottomUpOut", "bottomUpIn");

    net.initialize();

    ///////////////////////////////////////////////////////////////
    //
    //                 .------------------.
    //                 |    encoder       |
    //         data--->|  (RDSERegion)    |
    //                 |                  |
    //                 `------------------'
    //                     |           |
    //      .------------------.    .------------------.
    //      |    sp_local      |    |   sp_global      |
    //      |   (SPRegion)     |    |  (SPRegion)      |
    //      |                  |    |                  |
    //      `------------------'    `------------------'
    //                                       |
    //                              .------------------.
    //                              |      tm          |
    //                              |   (TMRegion)     |
    //                              |                  |
    //                              `------------------'
    //
    //////////////////////////////////////////////////////////////////


    // enable this to see a trace as it executes
    //net.setLogLevel(LogLevel::LogLevel_Verbose);

    std::cout << "Running: \n";
    // RUN
    for (size_t i = 0; i < EPOCHS; i++) {
      // genarate some data to send to the encoder
      //  -- A sin wave, one degree rotation per iteration, 1% noise added
      double data = std::sin(i * (3.1415 / 180)) + (double)rnd.realRange(-0.01f, +0.1f);
      encoder->setParameterReal64("sensedValue", data); // feed data into RDSE encoder for this iteration.

      // Execute an iteration.
      net.run(1);

      // output values
      float an = ((float *)tm->getOutputData("anomaly").getBuffer())[0];
      VERBOSE << "Epoch = " << i << std::endl;
      VERBOSE << "  Data        = " << data << std::endl;
      VERBOSE << "  Encoder out = " << encoder->getOutputData("encoded").getSDR();
//      VERBOSE << "  SP (local)  = " << sp_local->getOutputData("bottomUpOut").getSDR();
      VERBOSE << "  SP (global) = " << sp_global->getOutputData("bottomUpOut").getSDR();
      VERBOSE << "  TM output   = " << tm->getOutputData("bottomUpOut").getSDR();
      VERBOSE << "  ActiveCells = " << tm->getOutputData("activeCells").getSDR();
      VERBOSE << "  winners     = " << tm->getOutputData("predictedActiveCells").getSDR();
      VERBOSE << "  Anomaly     = " << an << std::endl;

      // Save the data for plotting.   <iteration>, <sin data>, <anomaly>\n
      if (ofs.is_open())
        ofs << i << "," << data << "," << an << std::endl;
    }
    if (ofs.is_open())
      ofs.close();
    std::cout << "finished\n";


  } catch (Exception &e) {
    std::cerr << e.what();
    if (ofs.is_open())
      ofs.close();
    return 1;
  }
  
  return 0;
}

