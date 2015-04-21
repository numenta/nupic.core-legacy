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

template<typename TraceType>
class Metric
{
public:
  // A metric computed over a set of data (usually from a `CountsTrace`).

  Instance* _monitor;
  string _title;

  Real _min, _max;
  Real _sum, _mean;
  Real _standardDeviation;

  vector<TraceType>* _data;

  Metric<TraceType>()
  {
    string emptyTitle("");
    _monitor = NULL;
    _title = emptyTitle;
  }
  
  Metric<TraceType>(Instance* monitor, string& title, vector<TraceType>* data);

  string prettyPrintTitle();

  void _computeStats();

  static Metric<TraceType> createFromTrace(Trace<TraceType>& trace, bool excludeResets = false);
  static Metric<TraceType> copy(const Metric<TraceType>& rhs);

};


class MetricsTrace : public Trace<Metric<int>>
{
public:
  // Each entry contains Metrics(for example metric for # of predicted = > active cells).
  string prettyPrintDatum(Metric<int>& datum);

};

#endif // NTA_metric_HPP
