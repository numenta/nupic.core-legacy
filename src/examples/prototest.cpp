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

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/utils/Random.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;

void testSP()
{
  Random random(10);

  const UInt inputSize = 500;
  const UInt numColumns = 500;
  const UInt w = 50;

  vector<UInt> inputDims{inputSize};
  vector<UInt> colDims{numColumns};

  SpatialPooler sp1;
  sp1.initialize(inputDims, colDims);

  UInt input[inputSize];
  for (UInt i = 0; i < inputSize; ++i)
  {
    if (i < w)
    {
      input[i] = 1;
    } else {
      input[i] = 0;
    }
  }
  UInt output[numColumns];

  for (UInt i = 0; i < 10000; ++i)
  {
    random.shuffle(input, input + inputSize);
    sp1.compute(input, true, output);
  }

  // Now we reuse the last input to test after serialization

  vector<UInt> activeColumnsBefore;
  for (UInt i = 0; i < numColumns; ++i)
  {
    if (output[i] == 1)
    {
      activeColumnsBefore.push_back(i);
    }
  }

  // Save initial trained model
  ofstream osA("outA.proto", ofstream::binary);
  sp1.write(osA);
  osA.close();

  ofstream osC("outC.proto", ofstream::binary);
  sp1.save(osC);
  osC.close();

  SpatialPooler sp2;

  Real64 timeA = 0.0, timeC = 0.0;

  for (UInt i = 0; i < 100; ++i)
  {
    // Create new input
    random.shuffle(input, input + inputSize);

    // Get expected output
    UInt outputBaseline[numColumns];
    sp1.compute(input, true, outputBaseline);

    // A - First do iostream version
    UInt outputA[numColumns];
    {
      SpatialPooler spTemp;

      nupic::Timer testTimer;
      testTimer.start();

      // Deserialize
      ifstream is("outA.proto", ifstream::binary);
      spTemp.read(is);
      is.close();

      // Feed new record through
      spTemp.compute(input, true, outputA);

      // Serialize
      ofstream os("outA.proto", ofstream::binary);
      spTemp.write(os);
      os.close();

      testTimer.stop();
      timeA = timeA + testTimer.getElapsed();
    }

    for (UInt i = 0; i < numColumns; ++i)
    {
      NTA_CHECK(outputBaseline[i] == outputA[i]);
    }

    // C - Next do old version
    UInt outputC[numColumns];
    {
      SpatialPooler spTemp;

      nupic::Timer testTimer;
      testTimer.start();

      // Deserialize
      ifstream is("outC.proto", ifstream::binary);
      spTemp.load(is);
      is.close();

      // Feed new record through
      spTemp.compute(input, true, outputC);

      // Serialize
      ofstream os("outC.proto", ofstream::binary);
      spTemp.save(os);
      os.close();

      testTimer.stop();
      timeC = timeC + testTimer.getElapsed();
    }

    for (UInt i = 0; i < numColumns; ++i)
    {
      NTA_CHECK(outputBaseline[i] == outputC[i]);
    }

  }

  remove("outA.proto");
  remove("outC.proto");

  cout << "Timing for SpatialPooler serialization (smaller is better):" << endl;
  cout << "Cap'n Proto: " << timeA << endl;
  cout << "Manual: " << timeC << endl;
}

void testRandomIOStream(UInt n)
{
  Random r1(7);
  Random r2;

  nupic::Timer testTimer;
  testTimer.start();
  for (UInt i = 0; i < n; ++i)
  {
    r1.getUInt32();

    // Serialize
    ofstream os("random2.proto", ofstream::binary);
    r1.write(os);
    os.flush();
    os.close();

    // Deserialize
    ifstream is("random2.proto", ifstream::binary);
    r2.read(is);
    is.close();

    // Test
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
  }
  testTimer.stop();

  remove("random2.proto");

  cout << "Cap'n Proto: " << testTimer.getElapsed() << endl;
}

void testRandomManual(UInt n)
{
  Random r1(7);
  Random r2;

  nupic::Timer testTimer;
  testTimer.start();
  for (UInt i = 0; i < n; ++i)
  {
    r1.getUInt32();

    // Serialize
    ofstream os("random3.proto", ofstream::binary);
    os << r1;
    os.flush();
    os.close();

    // Deserialize
    ifstream is("random3.proto", ifstream::binary);
    is >> r2;
    is.close();

    // Test
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
    NTA_CHECK(r1.getUInt32() == r2.getUInt32());
  }
  testTimer.stop();

  remove("random3.proto");

  cout << "Manual: " << testTimer.getElapsed() << endl;
}

int main(int argc, const char * argv[])
{
  UInt n = 1000;
  cout << "Timing for Random serialization (smaller is better):" << endl;
  testRandomIOStream(n);
  testRandomManual(n);

  testSP();

  return 0;
}
