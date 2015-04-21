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
#include "MetricTest.hpp"

using namespace std;

namespace nupic {

  void MetricTest::RunTests()
  {
    setup();

    testCreateFromTrace();
    testCreateFromTraceExcludeResets();
  }

  void MetricTest::setup()
  {
    trace = Trace<vector<int>>(&temp, string("# active cells"));
    
    trace._data.resize(6);
    //trace._data.assign({ 1, 2, 3, 4, 5, 0 });
  }


  void MetricTest::testCreateFromTrace()
  {
    //Metric<vector<int>> metric = Metric<vector<int>>::createFromTrace(trace);
    //assertEqual(metric._title, trace._title);
    //assertEqual(metric.min, 0);
    //assertEqual(metric.max, 5);
    //assertEqual(metric.sum, 15);
    //assertEqual(metric.mean, 2.5);
    //assertEqual(metric.standardDeviation, 1.707825127659933);
  }

  void MetricTest::testCreateFromTraceExcludeResets()
  {
    //BoolsTrace resetTrace = BoolsTrace(_monitor, "resets");
    //resetTrace._data = vector<bool>{ true, false, false, true, false, false };
    //Metric<vector<int>> metric = Metric::createFromTrace(_trace, resetTrace);
    //assertEqual(metric.title, self.trace.title);
    //assertEqual(metric.min, 0);
    //assertEqual(metric.max, 5);
    //assertEqual(metric.sum, 10);
    //assertEqual(metric.mean, 2.5);
    //assertEqual(metric.standardDeviation, 1.8027756377319946);
  }

}; // of namespace nupic
