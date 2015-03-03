/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

#include <iostream>
#include <vector>
#include <algorithm>    // std::generate
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/algorithms/Cells4.hpp"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;
using namespace nupic::algorithms::Cells4;

// function generator:
int RandomNumber01 () { return (rand()%2); }

int main(int argc, const char * argv[])
{
const int DIM = 2048;
const int DIM_INPUT = 10000;

    vector<UInt> inputDim;
    inputDim.push_back(DIM_INPUT);
    inputDim.push_back(1);

    vector<UInt> colDim;
    colDim.push_back(DIM);
    colDim.push_back(1);

    // generate random input
    vector<UInt> input(DIM_INPUT);
    generate(input.begin(), input.end(), RandomNumber01);
    vector<UInt> outSP(DIM); // active array, output of SP/TP
    vector<UInt> outTP(DIM);   

//    cout << "input=" << input << endl;
 
    SpatialPooler sp(inputDim, colDim);
    Cells4 tp;
    tp.initialize(DIM);

    //run
    fill(outSP.begin(), outSP.end(), 0);
    sp.compute(input.data(), true, outSP.data());
    cout << "SP=" << outSP << endl;

    fill(outTP.begin(), outTP.end(), 0);
    Real rIn[DIM] = {};
    Real rOut[DIM] = {};

    cout << "TP:" << endl;

    for (int i=0; i< DIM; i++) {
      rIn[i] = (Real)(outSP[i]);
      cout << rIn[i];
      cout << rOut[i];
    }
    cout << "OK" << endl;

    tp.compute(rIn, rOut, true, true);
    cout << "TP=" << rOut << endl;

    return 0;
}
