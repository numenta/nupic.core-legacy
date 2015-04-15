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

#include <nupic\types\Types.hpp>
#include "Instance.hpp"
#include "Trace.hpp"
#include "Metric.hpp"


// Metric class used in monitor mixin framework.

template<typename TraceType>
Metric<TraceType>::Metric(Instance* monitor, string& title, vector<TraceType>& data)
{
  //@param monitor(MonitorMixinBase) Monitor Mixin instance that generated this trace
  //@param title(string)           Title
  //@param data(list)             List of numbers to compute metric from
  _monitor = monitor;
  _title = title;
  _min = _max = _sum = 0.0;
  _mean = _standardDeviation = 0.0;

  _computeStats(data);
}

template<typename TraceType>
Metric<TraceType> createFromTrace(Trace<TraceType>& trace, bool excludeResets = false)
{
  vector<TraceType> data = list(trace.data);
  if (excludeResets)
    data = [x for i, x in enumerate(trace.data) if not excludeResets.data[i]];
  return Metric(trace.monitor, trace.title, data);
}

template<typename TraceType>
Metric<TraceType> Metric<TraceType>::copy(const Trace<TraceType>& rhs)
{
  Metric<TraceType> metric = Metric<TraceType>(rhs._monitor, rhs._title);

  metric._min = _min;
  metric._max = _max;
  metric._sum = _sum;
  metric._mean = _mean;
  metric._standardDeviation = _standardDeviation;

  return metric;
}

template<typename TraceType>
string Metric<TraceType>::prettyPrintTitle()
{
  return "";// ("[{0}] {1}".format(self.monitor.mmName, self.title)
            //if self.monitor.mmName is not None else self.title)
}

template<typename TraceType>
void Metric<TraceType>::_computeStats(vector<TraceType>& data)
{
  if (data.size() == 0)
    return;

  //_min = min(data);
  //_max = max(data);
  //_sum = sum(data);
  //_mean = numpy.mean(data);
  //_standardDeviation = numpy.std(data);
}

string MetricsTrace::prettyPrintDatum(int& datum)
{
  return "";
  //"min: " {0:.2f} ", max: " {1:.2f} ", sum: "{2:.2f}", mean: "{3:.2f}", std dev: " {4:.2f} ";
  //datum.min, datum.max, datum.sum, datum.mean, datum.standardDeviation)
}
