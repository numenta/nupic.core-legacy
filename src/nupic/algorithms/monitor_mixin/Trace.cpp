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
#include <sstream>
#include <iostream>
#include <algorithm>

#include "Trace.hpp"


template<typename TraceType>
string Trace<TraceType>::prettyPrintTitle()
{
  string ret;

  if (_monitor && _monitor->mmName.size() > 0)
    ret = "[" + _monitor->mmName + "]";
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
  return out;
}

template<typename TraceType>
Trace<TraceType> Trace<TraceType>::makeCountsTrace()
{
  return Trace<TraceType>();
}

template<typename TraceType>
Trace<TraceType> Trace<TraceType>::makeCumCountsTrace()
{
  return Trace<TraceType>();
}

Trace<vector<UInt>> CountsTrace::makeCountsTrace()
{
  //@return (CountsTrace)A new Trace made up of counts of this trace's indices.
  string title = "# " + _title;
  CountsTrace trace;
  trace._monitor = _monitor;
  trace._title = title;

  for (vector<UInt> indicies : _data)
  {
    trace._data.push_back(indicies);
  }

  return trace;
}


Trace<vector<UInt>> CountsTrace::makeCumCountsTrace()
{
  //@return (CountsTrace)A new Trace made up of cumulative counts of this trace's indices.
  string title = "# (cumulative) " + _title;
  CountsTrace trace;
  trace._monitor = _monitor;
  trace._title = title;

  Trace<vector<UInt>> countsTrace = makeCountsTrace();
  trace._data.push_back(vector<UInt>{ accumulate(countsTrace) });
  return trace;
}


int CountsTrace::accumulate(Trace<vector<UInt>>& trace)
{
  int total = 0;

  for (vector<UInt>& entry : trace._data)
  {
    for (int item : entry)
      total += item;
  }

  return total;
}


string CountsTrace::prettyPrintDatum(vector<UInt>& datum)
{
  stringstream ret;

  sort(datum.begin(), datum.end());

  for (int item : datum)
  {
    ret << item << ",";
  }

  return ret.str();
}

string BoolsTrace::prettyPrintDatum(vector<bool>& datum)
{
  stringstream ret;

  for (bool item : datum)
  {
    if (item)
      ret << "true,";
    else
      ret << "false,";
  }

  return ret.str();
}

string StringsTrace::prettyPrintDatum(vector<string>& datum)
{
  stringstream ret;

  for (string item : datum)
  {
    ret << item << ",";
  }

  return ret.str();
}
