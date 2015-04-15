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
* Implementation of TM mixin that enables detailed monitoring of history.
*/

#include <map>
#include <string>
#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>

#include <nupic\types\Types.hpp>
#include "Trace.hpp"
#include "Metric.hpp"


template<typename TraceType>
Trace<TraceType>::Trace(Instance& monitor, string& title)
{
  //@param monitor(MonitorMixinBase) Monitor Mixin instance that generated this trace
  //@param title(string)             Title
  _monitor = monitor;
  _title = title;
}

template<typename TraceType>
string Trace<TraceType>::prettyPrintTitle()
{
  string ret;

  if (_monitor.mmName.size() > 0)
    ret = "[" + _monitor.mmName + "]";
  else
    ret = "[" + _title + "]";

  ret += "{" + _title + "}";

  return ret;
}

template<typename TraceType>
string Trace<TraceType>::prettyPrintDatum(TraceType& datum)
{
  //@param datum(object) Datum from `self.data` to pretty - print
  //@return (string)Pretty - printed datum
  string out = "";

  //if (datum.size())
  //{
  //  for (auto data : datum)
  //    out += data.ToString();
  //  return out;
  //}

  return out;
}


template<typename TraceType>
Trace<TraceType> IndicesTrace::makeCountsTrace()
{
  //@return (CountsTrace)A new Trace made up of counts of this trace's indices.
  string title = "# " + _title;
  CountsTrace trace(_monitor, title);

  for (int indicies : _data)
  {
    trace._data.push_back(indicies);
  }
  return trace;
}

int IndicesTrace::accumulate(Trace<vector<int>>& trace)
{
  int total = 0;
  for (int item : trace._data)
    total += item;

  return total;
}

template<typename TraceType>
Trace<TraceType> IndicesTrace::makeCumCountsTrace()
{
  //@return (CountsTrace)A new Trace made up of cumulative counts of this trace's indices.
  string title = "# (cumulative) " + _title;
  CountsTrace trace = CountsTrace(_monitor, title);
  Trace<vector<int>> countsTrace = makeCountsTrace();

  //trace.data = list(accumulate(countsTrace.data));
  return trace;
}

string IndicesTrace::prettyPrintDatum(vector<int>& datum)
{
  return "";// str(sorted(list(datum)));
}
