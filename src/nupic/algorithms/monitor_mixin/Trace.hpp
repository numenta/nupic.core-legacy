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

#include "Instance.hpp"

namespace nupic
{
  template<typename TraceType>
  class Trace
  {
  public:
    Instance* _monitor;
    string _title;

    vector<TraceType> _data;

  public:
    Trace();
    Trace(Instance* monitor, string& title);

    virtual Trace<TraceType> makeCountsTrace();
    virtual Trace<TraceType> makeCumCountsTrace();

    virtual string prettyPrintTitle();
    virtual string prettyPrintDatum(TraceType& datum);
  };


  class CountsTrace : public Trace<vector<int>>
  {
  public:
    // Each entry contains counts(for example # of predicted = > active cells).
    CountsTrace() { };

  };


  class IndicesTrace : public CountsTrace
  {
  private:
    static int accumulate(Trace<vector<int>>& trace);

  public:
    virtual Trace<vector<int>> makeCountsTrace();
    virtual Trace<vector<int>> makeCumCountsTrace();

    virtual string prettyPrintDatum(vector<int>& datum);

  };


  class BoolsTrace : public Trace<vector<bool>>
  {
  public:
    // Each entry contains bools(for example resets).
    BoolsTrace() { };

  };


  class StringsTrace : public Trace<vector<string>>
  {
  public:
    // Each entry contains strings(for example sequence labels).
    StringsTrace() { };

  };

}; // of namespace nupic

#endif // NTA_trace_classes_HPP
