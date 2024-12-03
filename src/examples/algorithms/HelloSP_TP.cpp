/*
 * Copyright 2013-2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <algorithm> // std::generate
#include <cmath>     // pow
#include <cstdlib>   // std::rand, std::srand
#include <ctime>     // std::time
#include <iostream>
#include <vector>

#include "nupic/algorithms/Cells4.hpp"
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/os/Timer.hpp"

using namespace std;
using namespace nupic;
using nupic::algorithms::Cells4::Cells4;
using nupic::algorithms::spatial_pooler::SpatialPooler;

// function generator:
int RandomNumber01() {
  return (rand() % 2);
} // returns random (binary) numbers from {0,1}

int main(int argc, const char *argv[]) {
  const UInt DIM = 2048; // number of columns in SP, TP
  const UInt DIM_INPUT = 10000;
  const UInt TP_CELLS_PER_COL = 10; // cells per column in TP
  const UInt EPOCHS =
      pow(10, 4); // number of iterations (calls to SP/TP compute() )

  vector<UInt> inputDim = {DIM_INPUT};
  vector<UInt> colDim = {DIM};

  // generate random input
  vector<UInt> input(DIM_INPUT);
  vector<UInt> outSP(DIM); // active array, output of SP/TP
  const int _CELLS = DIM * TP_CELLS_PER_COL;
  vector<UInt> outTP(_CELLS);
  Real rIn[DIM] = {}; // input for TP (must be Reals)
  Real rOut[_CELLS] = {};

  // initialize SP, TP
  SpatialPooler sp(inputDim, colDim);
  Cells4 tp(DIM, TP_CELLS_PER_COL, 12, 8, 15, 5, .5, .8, 1.0, .1, .1, 0.0,
            false, 42, true, false);

  // Start a stopwatch timer
  Timer stopwatch(true);

  // run
  for (UInt e = 0; e < EPOCHS; e++) {
    generate(input.begin(), input.end(), RandomNumber01);
    fill(outSP.begin(), outSP.end(), 0);
    sp.compute(input.data(), true, outSP.data());
    sp.stripUnlearnedColumns(outSP.data());

    for (UInt i = 0; i < DIM; i++) {
      rIn[i] = (Real)(outSP[i]);
    }

    tp.compute(rIn, rOut, true, true);

    for (UInt i = 0; i < _CELLS; i++) {
      outTP[i] = (UInt)rOut[i];
    }

    // print
    if (e == EPOCHS - 1) {
      cout << "Epoch = " << e << endl;
      cout << "SP=" << outSP << endl;
      cout << "TP=" << outTP << endl;
    }
  }

  stopwatch.stop();
  cout << "Total elapsed time = " << stopwatch.getElapsed() << " seconds"
       << endl;

  return 0;
}
