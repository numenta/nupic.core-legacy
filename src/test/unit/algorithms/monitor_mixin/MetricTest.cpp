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
#include <nupic/math/Math.hpp>
#include "MetricTest.hpp"

using namespace std;
using namespace nupic;

void MetricTest::RunTests()
{
  setup();

  testCreateFromTrace();
  testCreateFromTraceExcludeResets();
}

void MetricTest::setup()
{
  trace._monitor = &temp;
  trace._title = string("# active cells");
    
  //trace._data.resize(6);
  trace._data.push_back({ 1, 2, 3, 4, 5, 0 });
};

void MetricTest::testCreateFromTrace()
{
  MetricsVector metric;
  metric.createFromTrace(trace);
    
  NTA_CHECK(metric._title == trace._title);
  NTA_CHECK(metric._min == 0);
  NTA_CHECK(metric._max == 5);
  NTA_CHECK(metric._sum == 15);
  NTA_CHECK(metric._mean == 2.5);
  bool eq = nupic::nearlyEqual(metric._standardDeviation, Real(1.707825127659933));
  EXPECT_TRUE(eq);
}

void MetricTest::testCreateFromTraceExcludeResets()
{
  vector<UInt> resetsList({ 1,0,0,1,0,0 });

  Trace<vector<UInt>> resetsTrace = Trace<vector<UInt>>(&temp, string("resets"));
  resetsTrace._data.push_back(resetsList);

  MetricsVector metric;
  metric.createFromTrace(trace, resetsTrace);

  NTA_CHECK(metric._title == trace._title);
  NTA_CHECK(metric._min == 0);
  NTA_CHECK(metric._max == 5);
  NTA_CHECK(metric._sum == 10);
  NTA_CHECK(metric._mean == 2.5);
  bool eq = nupic::nearlyEqual(metric._standardDeviation, Real(1.707825127659933));
  EXPECT_TRUE(eq);
}
