#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <fast_cpp_csv_parser/csv.h>

#include "nupic/encoders/ScalarEncoder.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/algorithms/Cells4.hpp"
#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/os/Timer.hpp"
#include "nupic/utils/VectorHelpers.hpp"

using namespace std; // generic namespaces are included here
using namespace nupic;
using namespace nupic::utils;
using namespace nupic::algorithms::spatial_pooler;
using namespace nupic::algorithms::Cells4;
using namespace nupic::algorithms::anomaly;

const int DEBUG_LEVEL =1; //0=no debug, ..

class AnomalyDetection
{

  private:
    const UInt COLS = 2048;
    const UInt CELLS = 4;
    const Real INPUT_MIN = 0; 
    const Real INPUT_MAX = 100;
    // encoder
    ScalarEncoder encoder{
        25,   // w: Active bits per value
        INPUT_MIN,   // min value
        INPUT_MAX,  // max value
        0,  // n: bits per output value,
        0,   // radius
        0.1,   // resolution; only 1 of these 3 values can be set
        true // Clip input to min/max values if needed.
    };
    // Spatial Pooler
    SpatialPooler sp{
        std::vector<UInt>{(UInt)encoder.getOutputWidth()}, // Input dimensions (only one here)
        std::vector<UInt>{COLS}  // Number of colums per dimension
    };
    // Temporal Pooler
    Cells4 tp{
        COLS,    // Number of columns
        CELLS,     // Cells per column
        12,
        8,
        15,
        5,
        .5,
        .8,
        1.0,
        .1,
        .1,
        0.0,
        false,
        42,
        true,
        false //TODO use default params
    };
    // Anomaly
    Anomaly anomaly{2, AnomalyMode::PURE, 0};

    std::vector<UInt> lastTPOutput{COLS};

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
          VectorHelpers::print_vector(spOutput);
          std::cout << std::endl;
        }

        // The temporal pooler uses Real32, so the vector must be converted again
        auto tpInput = VectorHelpers::castVectorType<UInt, Real>(spOutput);
        std::vector<Real> tpOutput(tp.nCells());
        tp.compute(tpInput.data(), tpOutput.data(), true, true);

        // And the result is converted ONCE again to UInts for pretty printing to stdout.
        auto uintTpOutput = VectorHelpers::castVectorType<Real32, UInt>(tpOutput);
        if (DEBUG_LEVEL > 3) {
          std::cout << "Output of temporal pooler: ";
          VectorHelpers::print_vector(uintTpOutput);
          std::cout << std::endl;
        }

        Real anScore = anomaly.compute(
          VectorHelpers::binaryToSparse<UInt>(spOutput),     // Spatial pooler (current active columns)
          VectorHelpers::binaryToSparse<UInt>(lastTPOutput), // Temporal pooler (previously predicted columns)
          0,
          0
        );
        // Save the output of the TP for the next iteration...
        lastTPOutput = VectorHelpers::cellsToColumns(uintTpOutput, CELLS);

        if(DEBUG_LEVEL > 0) {
          std::cout << "Input:\t" << x << "\tAnomaly: " << anScore << std::endl;
        } else if (DEBUG_LEVEL > 4) {
          std::cout << "Normalized TP Output: ";
          VectorHelpers::print_vector(lastTPOutput);
        }

        return anScore;
    }
};

int main()
{
    using namespace nupic;
    using namespace nupic::utils;

    // initialize
    // CSV I/O
    const UInt WORKING_COL = 2; // number of cols we want to read, in this example only 1st
    io::CSVReader<WORKING_COL> in(
               std::string(std::getenv("NUPIC_CORE"))
             + std::string("/src/examples/algorithms/hotgym.csv")); // CSV parser, see external/common/include/fast_cpp_csv_parser for usage.
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
    AnomalyDetection runner;
    // timer
    Timer stopwatch;

    // run through the data
    UInt iteration = 0;
    while (in.read_row(timestamp, consumption)) {
        // measure the HTM computation only (excluding time for CSV processing, disk I/O etc
        stopwatch.start();
        Real result = runner.compute(consumption); // do the anomaly computation
        stopwatch.stop();
        outFile << result << std::endl;
        ++iteration;
        if (DEBUG_LEVEL > 1) {
          std::cout << "--------------------------------------------------------\n\n";
        }
    }
    std::cout << "Total elapsed time = " << stopwatch.getElapsed() << " seconds" << std::endl;
    outFile.close();
    return 0;
}
