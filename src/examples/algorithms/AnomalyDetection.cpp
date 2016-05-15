/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

/*
 This file is a low-level (using directly Encoder, SpatialPooler, 
 TemporalPooler/Memory and Anomaly) example of anomaly detection 
 using nupic.core (pure C++). Its purpose is to serve as a working
 example similar to hotgym from python repo. 

 The file is separated into 2 parts: 
 :AnomalyDetection class: 
   Provides the HTM anomaly detection functionality; 
   parametric constructor lets us set the most useful parameters.
   compute() which returns anomaly score for given input.
 :main():
   Responsible for parsing CSV input files, calling an AnomalyDetection
   instance and writing results to a file. 
   Optionally se DEBUG_LEVEL to see desired details. 
   Available is also functionality to benchmark performance of the anomaly
   detection using provided stopwatch Timer. 

 Usage:
   run $RELEASE/bin/anomaly_example 
 */

#include <iostream>
#include <fstream> // for file output
#include <vector>
#include <string>
#include <cstdlib>

#include "nupic/encoders/ScalarEncoder.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/algorithms/Cells4.hpp" // TP
#include <nupic/algorithms/TemporalMemory.hpp> // TM
#include "nupic/algorithms/Anomaly.hpp"

#include <csv.h> // external CSV parser
#include "nupic/os/Timer.hpp"
#include "VectorHelpers.hpp"

using namespace std; // generic namespaces are included here
using namespace nupic;
using namespace nupic::utils;
using namespace nupic::algorithms::spatial_pooler;
using namespace nupic::algorithms::Cells4;
using namespace nupic::algorithms::temporal_memory;
using namespace nupic::algorithms::anomaly;

const int DEBUG_LEVEL =1; //0=no debug (also disabled timer), ..

class AnomalyDetection
{

  private:
    ScalarEncoder encoder;
    SpatialPooler sp;
    Cells4 tp;
    TemporalMemory tm;
    Anomaly anomaly; 
    std::vector<UInt> lastTPOutput_;
    const string tmImpl; //"TM","TP"


  public:
    Real compute(Real x) {
        auto scalar = std::vector<Real32>(encoder.getOutputWidth());
        encoder.encodeIntoArray(x, scalar.data());

        // For some reason, the ScalarEncoder outputs a Real32 vector,
        // which is incomaptible with the SpatialPooler's input, hence a new
        // vector is created with the required type, and the elements
        // are converted.
        auto uint_scalar = VectorHelpers::castVectorType<Real32, UInt>(scalar);
        if (DEBUG_LEVEL > 1) {
          std::cout << "Scalar encoder: ";
          VectorHelpers::print_vector(uint_scalar);
        }

        std::vector<UInt> spOutput(sp.getNumColumns());
        sp.compute(uint_scalar.data(), true, spOutput.data());
        sp.stripUnlearnedColumns(spOutput.data());
        if (DEBUG_LEVEL > 2) {
          std::cout << "Spatial pooler: ";
          VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(spOutput), ",");
          std::cout << std::endl;
        }

        std::vector<UInt> tpOutput;
        if (tmImpl == "TP") {
          std::vector<Real> rTpOutput(tp.nCells());
          // The temporal pooler uses Real32, so the vector must be converted again
          auto tpInput = VectorHelpers::castVectorType<UInt, Real>(spOutput);
          tp.compute(tpInput.data(), rTpOutput.data(), true, true);
          // And the result is converted ONCE again to UInts
          tpOutput = VectorHelpers::castVectorType<Real32, UInt>(rTpOutput);
        } else {
          tm.compute(spOutput.size(), spOutput.data(), true);
          tpOutput = tm.getActiveCells(); //FIXME do union with getPredictedCells() , like TP.outputMode="both"
        }

        if (DEBUG_LEVEL > 3) {
          std::cout << "Output of temporal pooler: ";
          VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(tpOutput), ",");
          std::cout << std::endl;
        }

        Real anScore = anomaly.compute(
          VectorHelpers::binaryToSparse<UInt>(spOutput),     // Spatial pooler (current active columns)
          VectorHelpers::binaryToSparse<UInt>(lastTPOutput_), // Temporal pooler (previously predicted columns)
          0,
          0
        );
        // Save the output of the TP for the next iteration...
        if (tmImpl == "TP") {
          lastTPOutput_ = VectorHelpers::cellsToColumns(tpOutput, tp.nCellsPerCol());
        } else {
          lastTPOutput_ = VectorHelpers::cellsToColumns(tpOutput, tm.getCellsPerColumn());
        }

        if (DEBUG_LEVEL > 4) {
          std::cout << "Normalized TP Output: ";
          VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(lastTPOutput_), ",");
        }

        if(DEBUG_LEVEL > 0) {
          std::cout << "Input:\t" << x << "\tAnomaly: " << anScore << std::endl;
        }

        return anScore;
    }
    // constructor with default parameters
    AnomalyDetection(Real inputMin = 0.0, Real inputMax = 100.0, Real inputResolution = 0.1,
      UInt nCols = 2048, UInt nCells = 4, UInt anomalyWindowSize = 2, std::string tmImpl = "TM") : 
        encoder{25, inputMin, inputMax, 0, 0, inputResolution, false},
        sp{std::vector<UInt>{static_cast<UInt>(encoder.getOutputWidth())}, std::vector<UInt>{nCols}},
        tp{sp.getNumColumns(), nCells, 12, 8, 15, 5, .5, .8, 1.0, .1, .1, 0.0, false, 42, true, false}, //FIXME tp & tm should not be initialized both, but I dont know how to conditionally initialize based on impl.
        tm({sp.getNumColumns()}, nCells), //FIXME ensure same params TP/TM
        anomaly{anomalyWindowSize, AnomalyMode::PURE, 0},
        lastTPOutput_(nCols),
        tmImpl(tmImpl)
    {
    }
};

int main()
{
    // initialize
    // CSV I/O
    const int COUNT = 5000; //how many rows to process, -1 for whole file
    const UInt WORKING_COL = 2; // number of cols we want to read, in this example only 1st
    io::CSVReader<WORKING_COL> in(
               std::string(std::getenv("NUPIC_CORE"))
             + std::string("/src/examples/algorithms/hotgym.csv")); // CSV parser, see examples/CSV_README.md for usage.
    in.read_header(io::ignore_extra_column, "timestamp", "consumption"); //which column(s) we work on
    in.next_line(); in.next_line(); //skip the 2 meta rows in our header
    std::string timestamp;
    Real consumption;
    // write output to a file (don't forget to close later)
    std::ofstream outFile;
    outFile.open (
               std::string(std::getenv("NUPIC_CORE"))
             + std::string("/src/examples/algorithms/anomaly_out.csv"));
    outFile << "anomaly_score" << std::endl;

    // the running example class above
    AnomalyDetection runner {0.0, 55.0, 0.1, 2048, 8, 2}; //parameters; TODO optimize
    // timer
    Timer stopwatch;

    // run through the data
    int iteration = 0; //int, bcs compared with DEBUG (int) and Werror
    while ((iteration <= COUNT || COUNT < 0) && in.read_row(timestamp, consumption)) {
        // measure the HTM computation only (excluding time for CSV processing, disk I/O etc
        if (DEBUG_LEVEL > 0) {
          stopwatch.start();
        }
        Real result = runner.compute(consumption); // do the anomaly computation
        if (DEBUG_LEVEL > 0) {
          stopwatch.stop();
        }
        outFile << result << std::endl;
        ++iteration;
        if (DEBUG_LEVEL > 1) {
          std::cout << iteration << "--------------------------------------------------------\n\n";
        }
    }
    if (DEBUG_LEVEL > 0) {
      std::cout << "Total elapsed time = " << stopwatch.getElapsed() << " seconds" << std::endl;
    }
    outFile.close();
    return 0;
}
