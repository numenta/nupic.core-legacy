/* ---------------------------------------------------------------------
* Numenta Platform for Intelligent Computing (NuPIC)
* Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

/** @file
* Implementation of unit tests for Metric class
*/

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include "TraceTest.hpp"

using namespace std;

namespace nupic {

  void TraceTest::setUp()
  {
    //self.trace = IndicesTrace(self, "active cells")
    //self.trace.data.append(set([1, 2, 3]))
    //self.trace.data.append(set([4, 5]))
    //self.trace.data.append(set([6]))
    //self.trace.data.append(set([]))
  }

  void TraceTest::RunTests()
  {
    testMakeCountsTrace();
    testMakeCumCountsTrace();
  }

  void TraceTest::testMakeCountsTrace()
  {
    //countsTrace = self.trace.makeCountsTrace()
    //self.assertEqual(countsTrace.title, "# active cells")
    //self.assertEqual(countsTrace.data, [3, 2, 1, 0])
  }

  void TraceTest::testMakeCumCountsTrace()
  {
    //countsTrace = self.trace.makeCumCountsTrace()
    //self.assertEqual(countsTrace.title, "# (cumulative) active cells")
    //self.assertEqual(countsTrace.data, [3, 5, 6, 6])
  }

}; // of namespace nupic
