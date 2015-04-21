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

#include <string>
#include <iostream>
#include <sstream>
#include <boost/format.hpp>

#include <nupic\types\Types.hpp>
#include "Instance.hpp"
#include "Trace.hpp"
#include "Metric.hpp"

using namespace boost;


template<typename TraceType>
Metric<TraceType>::Metric(Instance* monitor, string& title, vector<TraceType>* data)
{
  //@param monitor(MonitorMixinBase)  Monitor Mixin instance that generated this trace
  //@param title(string)              Title
  //@param data(list)                 List of numbers to compute metric from
  _monitor = monitor;
  _title = title;

  _min = _max = _sum = 0.0;
  _mean = _standardDeviation = 0.0;

  _data = data;
  _computeStats(data);
}

template<typename TraceType>
Metric<TraceType> Metric<TraceType>::createFromTrace(Trace<TraceType>& trace, bool excludeResets)
{
  vector<TraceType> data = trace.data;

//  if (excludeResets)
//    data = [x for i, x in enumerate(trace.data) if not excludeResets.data[i]];

  return Metric<TraceType>(trace.monitor, trace.title, &data);
}

template<typename TraceType>
Metric<TraceType> Metric<TraceType>::copy(const Metric<TraceType>& rhs)
{
  Metric<TraceType> metric = Metric<TraceType>(rhs._monitor, rhs._title);

  metric._min = rhs._min;
  metric._max = rhs._max;
  metric._sum = rhs._sum;
  metric._mean = rhs._mean;
  metric._standardDeviation = rhs._standardDeviation;

  return metric;
}
template<typename TraceType>
Metric<TraceType> copy(const Metric<TraceType>& rhs)
{
  Metric<TraceType> metric = Metric<TraceType>(rhs._monitor, rhs._title);

  metric._min = rhs._min;
  metric._max = rhs._max;
  metric._sum = rhs._sum;
  metric._mean = rhs._mean;
  metric._standardDeviation = rhs._standardDeviation;

  return metric;
}

template<typename TraceType>
string Metric<TraceType>::prettyPrintTitle()
{
  return "";// ("[{0}] {1}".format(self.monitor.mmName, self.title)
            //if self.monitor.mmName is not None else self.title)
}

template<typename TraceType>
void Metric<TraceType>::_computeStats()
{
  if (_data.size() == 0)
    return;

  //_min = min(_data);
  //_max = max(_data);
  //_sum = sum(_data);
  //_mean = numpy.mean(_data);
  //_standardDeviation = numpy.std(_data);
}

string MetricsTrace::prettyPrintDatum(Metric<int>& datum)
{
  stringstream ss;

  ss << format("min: %.2fn") % datum._min;
  ss << format(", max: %.2fn") % datum._max;
  ss << format(", sum: %.2fn") % datum._sum;
  ss << format(", mean: %.2fn") % datum._mean;
  ss << format(", std dev: %.2fn") % datum._standardDeviation;
  return ss.str();
}
