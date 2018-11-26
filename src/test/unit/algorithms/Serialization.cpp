/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/utils/Random.hpp>

namespace testing {

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;

TEST(Serialization, testSP) {
  Random random(10);

  const UInt inputSize = 500;
  const UInt numColumns = 500;
  const UInt w = 50;

  vector<UInt> inputDims{inputSize};
  vector<UInt> colDims{numColumns};

  SpatialPooler sp1;
  sp1.initialize(inputDims, colDims);

  UInt input[inputSize];
  for (UInt i = 0; i < inputSize; ++i) {
    if (i < w) {
      input[i] = 1;
    } else {
      input[i] = 0;
    }
  }
  UInt output[numColumns];

  for (UInt i = 0; i < 10000; ++i) {
    random.shuffle(input, input + inputSize);
    sp1.compute(input, true, output);
  }

  // Now we reuse the last input to test after serialization

  vector<UInt> activeColumnsBefore;
  for (UInt i = 0; i < numColumns; ++i) {
    if (output[i] == 1) {
      activeColumnsBefore.push_back(i);
    }
  }

  // Save initial trained model
  ofstream osC("outC.stream", ofstream::binary);
  sp1.save(osC);
  osC.close();

  SpatialPooler sp2;

  for (UInt i = 0; i < 100; ++i) {
    // Create new input
    random.shuffle(input, input + inputSize);

    // Get expected output
    UInt outputBaseline[numColumns];
    sp1.compute(input, true, outputBaseline);

    // C - Next do old version
    UInt outputC[numColumns];
    {
      SpatialPooler spTemp;

      nupic::Timer testTimer;
      testTimer.start();

      // Deserialize
      ifstream is("outC.stream", ifstream::binary);
      spTemp.load(is);
      is.close();

      // Feed new record through
      spTemp.compute(input, true, outputC);

      // Serialize
      ofstream os("outC.stream", ofstream::binary);
      spTemp.save(os);
      os.close();

      testTimer.stop();
//      cout << "Timing for SpatialPooler serialization (smaller is better):" << endl;
//      cout << "Stream: " << testTimer.getElapsed() << endl;
    }

    for (UInt i = 0; i < numColumns; ++i) {
      ASSERT_EQ(outputBaseline[i], outputC[i]);
    }
  }

  remove("outC.stream");
}



TEST(serialization, testRandom) {
	const UInt n=1000;
  Random r1(7);
  Random r2;

  nupic::Timer testTimer;
  testTimer.start();
  for (UInt i = 0; i < n; ++i) {
    r1.getUInt32();

    // Serialize
    ofstream os("random3.stream", ofstream::binary);
    os << r1;
    os.flush();
    os.close();

    // Deserialize
    ifstream is("random3.stream", ifstream::binary);
    is >> r2;
    is.close();

    // Test
    ASSERT_EQ(r1.getUInt32(), r2.getUInt32());
    ASSERT_EQ(r1.getUInt32(), r2.getUInt32());
    ASSERT_EQ(r1.getUInt32(), r2.getUInt32());
    ASSERT_EQ(r1.getUInt32(), r2.getUInt32());
    ASSERT_EQ(r1.getUInt32(), r2.getUInt32());
  }
  testTimer.stop();

  remove("random3.stream");

  cout << "Random serialization: " << testTimer.getElapsed() << endl;
}

} //ns
