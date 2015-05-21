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
* Implementation of Metric class used in monitor mixin framework.
*/

#ifndef NTA_metric_HPP
#define NTA_metric_HPP

#include <map>
#include <string>

#include <nupic\types\Types.hpp>
#include "Instance.hpp"
#include "Trace.hpp"

namespace nupic
{
  template<typename TraceType>
  class Metric
  {
  public:
    // A metric computed over a set of data (usually from a `CountsTrace`).

    Instance _monitor;
    string _title;

    Real _min, _max;
    Real _sum, _mean;
    Real _standardDeviation;

    vector<TraceType> _data;

    Metric()
    {
      string emptyTitle("");
      _title = emptyTitle;
    }

    Metric(Instance& monitor, string& title, TraceType& data);

    string prettyPrintTitle();

    virtual void _computeStats(Trace<vector<UInt>>& resets);

    virtual Metric<TraceType> createFromTrace(Trace<TraceType>& trace);
    virtual Metric<TraceType> createFromTrace(Trace<TraceType>& trace, Trace<vector<UInt>>& resets);

    static Metric<TraceType> copy(const Metric<TraceType>& rhs);

  };

  class MetricsVector : public Metric<vector<UInt>>
  {
  public:
    virtual void _computeStats(Trace<vector<UInt>>& resets);

    static Metric<vector<UInt>> createFromTrace(Trace<vector<UInt>>& trace);
    static Metric<vector<UInt>> createFromTrace(Trace<vector<UInt>>& trace, Trace<vector<UInt>>& resets);

  };

  class MetricsTrace : public Trace<Metric<UInt>>
  {
  public:
    // Each entry contains Metrics(for example metric for # of predicted = > active cells).
    string prettyPrintDatum(Metric<int>& datum);

  };

}; // of namespace nupic

#endif // NTA_metric_HPP
