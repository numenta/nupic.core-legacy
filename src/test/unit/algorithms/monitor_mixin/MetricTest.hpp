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
* Definitions for unit tests for Metric class
*/

#ifndef NTA_METRIC_TEST
#define NTA_METRIC_TEST

#include <cstring>
#include <fstream>
#include <stdio.h>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/test/Tester.hpp>
#include "Metric.hpp"

using namespace std;

namespace nupic {

  class MetricTest : public Tester
  {
  public:
    MetricTest() {}
    virtual ~MetricTest() {}

    // Run all appropriate tests
    virtual void RunTests() override;

  private:
    //Trace<vector<int>> trace;

    void setup()
    {
      //trace = Trace<vector<int>>(_monitor, "# active cells");
      //trace._data = vector<int>{ 1, 2, 3, 4, 5, 0 };
    }


    void testCreateFromTrace()
    {
      //Metric<vector<int>> metric = createFromTrace(trace);
      //assertEqual(metric.title, self.trace.title);
      //assertEqual(metric.min, 0);
      //assertEqual(metric.max, 5);
      //assertEqual(metric.sum, 15);
      //assertEqual(metric.mean, 2.5);
      //assertEqual(metric.standardDeviation, 1.707825127659933);
    }


    void testCreateFromTraceExcludeResets()
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

  }; // end class MetricTest

} // end namespace nupic

#endif // NTA_METRIC_TEST
