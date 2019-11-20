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

  //  Runtime arguments:  hotgym_napi [epochs [filename]]
  if(argc >= 2) {
    EPOCHS = std::stoi(argv[1]);         // number of iterations (default 5000)
  }
  if (argc >= 3) {
    ofs.open(argv[2], std::ios::out);    // output filename (for plotting)
  }

  try {

    std::cout << "initializing\n";

    Network net;

    // Declare the regions to use
    std::string encoder_parameters = "{size: " + std::to_string(DIM_INPUT) + ", activeBits: 4, radius: 0.5, seed: 2019 }";
    std::shared_ptr<Region> region1 =  net.addRegion("region1", "RDSERegion", encoder_parameters);
    std::shared_ptr<Region> region2a = net.addRegion("region2a", "SPRegion", "{columnCount: " + std::to_string(COLS) + "}");
    std::shared_ptr<Region> region2b = net.addRegion("region2b", "SPRegion", "{columnCount: " + std::to_string(COLS) + ", globalInhibition: true}");
    std::shared_ptr<Region> region3 =  net.addRegion("region3", "TMRegion", "{activationThreshold: 9, cellsPerColumn: " + std::to_string(CELLS) + "}");

    // Setup data flows between regions
    net.link("region1", "region2a", "", "", "encoded", "bottomUpIn");
    net.link("region1", "region2b", "", "", "encoded", "bottomUpIn");
    net.link("region2b", "region3", "", "", "bottomUpOut", "bottomUpIn");

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
    //      |   SP (local)     |    |   SP (global)    |
    //      |                  |    |                  |
    //      |                  |    |                  |
    //      `------------------'    `------------------'
    //                                       |
    //                              .------------------.
    //                              |      TM          |
    //                              |   (TMRegion)     |
    //                              |                  |
    //                              `------------------'
    //
    //////////////////////////////////////////////////////////////////


    // enable this to see a trace as it executes
    // net.setLogLevel(LogLevel::LogLevel_Verbose);

    std::cout << "Running: \n";
    // RUN
    for (size_t i = 0; i < EPOCHS; i++) {
      // genarate some data to send to the encoder
      //  -- A sin wave, one degree rotation per iteration, 1% noise added
      double data = std::sin(i * (3.1415 / 180)) + (double)rnd.realRange(-0.01f, +0.1f);
      region1->setParameterReal64("sensedValue", data); // feed data into RDSE encoder for this iteration.

      // Execute an iteration.
      net.run(1);

      // output values
      double an = ((double *)region3->getOutputData("anomaly").getBuffer())[0];
      VERBOSE << "Epoch = " << i << std::endl;
      VERBOSE << "  Data        = " << data << std::endl;
      VERBOSE << "  Encoder out = " << region1->getOutputData("encoded").getSDR() << std::endl;
      VERBOSE << "  SP (local)  = " << region2a->getOutputData("bottomUpOut").getSDR() << std::endl;
      VERBOSE << "  SP (global) = " << region2b->getOutputData("bottomUpOut").getSDR() << std::endl;
      VERBOSE << "  TM output   = " << region3->getOutputData("bottomUpOut").getSDR() << std::endl;
      VERBOSE << "  ActiveCells = " << region3->getOutputData("activeCells").getSDR() << std::endl;
      VERBOSE << "  winners     = " << region3->getOutputData("predictedActiveCells").getSDR() << std::endl;
      VERBOSE << "  Anomaly     = " << an << std::endl;

      // Save the data for plotting.   <iteration>, <sin data>, <anomaly>\n
      if (ofs.is_open())
        ofs << i << "," << data << "," << an << std::endl;
    }
    if (ofs.is_open())
      ofs.close();

  } catch (Exception &e) {
    std::cerr << e.what();
    if (ofs.is_open())
      ofs.close();
    return 1;
  }
  
  return 0;
}

