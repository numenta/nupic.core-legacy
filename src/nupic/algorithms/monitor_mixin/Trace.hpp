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
    Trace()
      : _monitor(NULL), _title("") {};

    Trace(Instance* monitor, string& title)
      : _monitor(monitor), _title(title) {};

    virtual Trace<TraceType> makeCountsTrace();
    virtual Trace<TraceType> makeCumCountsTrace();

    string prettyPrintTitle();

    virtual string prettyPrintDatum(TraceType& datum);
  };

  class CountsTrace : public Trace<vector<UInt>>
  {
  private:
    static UInt accumulate(Trace<vector<UInt>>& trace);

  public:
    // Each entry contains counts(for example # of predicted = > active cells).
    CountsTrace() {};
    CountsTrace(Instance* monitor, string& title)
    {
      _monitor = monitor;
      _title = title;
    };

    virtual Trace<vector<UInt>> makeCountsTrace();
    virtual Trace<vector<UInt>> makeCumCountsTrace();

    virtual string prettyPrintDatum(vector<UInt>& datum);

  };

  class IndicesTrace : public CountsTrace
  {
  public:
    IndicesTrace() {};
    IndicesTrace(Instance* monitor, string& title)
    {
      _monitor = monitor;
      _title = title;
    };

  };

  class BoolsTrace : public Trace<vector<bool>>
  {
  public:
    // Each entry contains bools(for example resets).
    BoolsTrace() {};
    BoolsTrace(Instance* monitor, string& title)
    {
      _monitor = monitor;
      _title = title;
    };

    virtual string prettyPrintDatum(vector<bool>& datum);
  };

  class StringsTrace : public Trace<string>
  {
  public:
    // Each entry contains strings(for example sequence labels).
    StringsTrace() {};
    StringsTrace(Instance* monitor, string& title)
    {
      _monitor = monitor;
      _title = title;
    };

    virtual string prettyPrintDatum(vector<string>& datum);
  };
 }; // of namespace nupic
#endif // NTA_trace_classes_HPP
