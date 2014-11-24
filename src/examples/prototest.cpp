/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <nupic/utils/Random.hpp>

using namespace std;
using namespace nupic;

long diff(timeval & start, timeval & end)
{
  return (
      ((end.tv_sec - start.tv_sec) * 1000000) +
      (end.tv_usec - start.tv_usec)
  );
}

void testRandomIOStream()
{
  Random r1(7);
  Random r2;

  time_t startStream, endStream;
  time(&startStream);
  for (UInt i = 0; i < 50000; ++i)
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
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
  }
  time(&endStream);
  cout << "Stream time: " << (endStream - startStream) << endl;
}

void testRandomManual()
{
  Random r1(7);
  Random r2;

  time_t startManual, endManual;
  time(&startManual);
  for (UInt i = 0; i < 50000; ++i)
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
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
    NTA_ASSERT(r1.getUInt32() == r2.getUInt32());
  }
  time(&endManual);
  cout << "Manual time: " << (endManual - startManual) << endl;
}

int main(int argc, const char * argv[])
{
  testRandomIOStream();
  testRandomManual();

  return 0;
}
