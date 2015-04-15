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
* ----------------------------------------------------------------------
*/

/** @file
* Definitions for A record of the past data the algorithm has seen, 
* with an entry for each iteration.
*/

#ifndef NTA_trace_classes_HPP
#define NTA_trace_classes_HPP

#include <vector>
#include <nupic\types\Types.hpp>
#include "Instance.hpp"

using namespace std;
using namespace nupic;

template<typename TraceType>
class Trace
{
protected:
  Instance* _monitor;
  string _title;

  TraceType& _data;

public:
  Trace(Instance* monitor, string& title);

  virtual string prettyPrintTitle();
  virtual string prettyPrintDatum(TraceType& datum);

};

class IndicesTrace : public Trace<vector<int>>
{
private:
  int accumulate(Trace<vector<int>>& iterator);

public:
  Trace<vector<int>> makeCountsTrace();
  Trace<vector<int>> makeCumCountsTrace();

  string prettyPrintDatum(vector<int>& datum);

};


class BoolsTrace : public Trace<vector<bool>>
{
  // Each entry contains bools(for example resets).

};


class CountsTrace : public Trace<vector<CountsTrace>>
{
  // Each entry contains counts(for example # of predicted = > active cells).
};


class StringsTrace : public Trace<vector<string>>
{
  // Each entry contains strings(for example sequence labels).
};

#endif // NTA_trace_classes_HPP
