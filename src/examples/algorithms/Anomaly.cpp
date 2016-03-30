#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <cstdlib>

#include "nupic/encoders/ScalarEncoder.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/algorithms/Cells4.hpp"
#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/os/Timer.hpp"
#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/CSVHelpers.hpp"

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
        if (DEBUG_LEVEL > 0) 
        {
          std::cout << x << ":\n";
        } else if (DEBUG_LEVEL > 1) {
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
        std::cout << "Anomaly: "
                  << anScore << "\n\n";
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
    using namespace nupic::utils::csv;

    // initialize
    // CSV I/O
    const UInt HEADER_ROWS = 4; //will be skipped
    const UInt WORKING_COL = 0; // in this example we work on 1st column
    CSVReader<Real> reader(
            std::string(std::getenv("NUPIC_CORE"))
            + std::string("/src/examples/algorithms/hotgym.csv"),
            HEADER_ROWS); //runs the temporal/hotgym data

    CSVWriter<Real> writer(
            std::string(std::getenv("NUPIC_CORE"))
            + std::string("/src/examples/algorithms/anomaly_out.csv"));

    if (DEBUG_LEVEL > 0) {
      std::vector<std::string> column1 = reader.readColumn(WORKING_COL);
      std::cout << "Whole input: ";
      VectorHelpers::print_vector(column1, " ");
      std::cout << std::endl;
    }
    // the running example class above
    AnomalyDetection runner;
    // timer
    Timer stopwatch;

    // run through the data
    UInt iteration = 0;
    vector<string> row = {};
    while ((row = reader.getLine()).size() > 0) {
        auto elem = stof(row[WORKING_COL]);
        // measure the HTM computation only (excluding time for CSV processing, disk I/O etc
        stopwatch.start();
        Real result = runner.compute(elem); // do the anomaly computation
        stopwatch.stop();
        writer.writeLine(std::vector<Real>{result});
        ++iteration;
        if (DEBUG_LEVEL > 1) {
          std::cout << "--------------------------------------------------------\n\n";
        }
    }
    std::cout << "Total elapsed time = " << stopwatch.getElapsed() << " seconds" << std::endl;
    return 0;
}
