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
  // A metric computed over a set of data(usually from a `CountsTrace`).
  Instance* _monitor;
  string& _title;

  Real _min, _max, _sum, _mean, _standardDeviation;

  vector<TraceType> data;

  Metric(Instance* monitor, string& title, vector<TraceType>& data);

  Metric<TraceType> copy(const Trace<TraceType>& rhs);

  string prettyPrintTitle();

  void _computeStats(vector<TraceType>& data);
};


class MetricsTrace : public Trace<vector<Metric<int>&>>
{
  // Each entry contains Metrics(for example metric for # of predicted = > active cells).
  string prettyPrintDatum(int& datum);

};

#endif // NTA_metric_HPP
