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

#ifndef NTA_MONITOR_MIXIN_HPP
#define NTA_MONITOR_MIXIN_HPP

/** @file
* Definition of MonitorMixinBase class used in
* monitor mixin framework.
*/

#include <map>
#include <string>

#include <nupic\types\Types.hpp>

#include "Trace.hpp"
#include "Metric.hpp"


class MonitorMixinBase
{
protected:
  string _mmName;

  map<string, Trace<vector<int>>> _mmTraces;

public:
  // Base class for MonitorMixin.
  // Each subclass will be a mixin for a particular algorithm.
  // 
  // All arguments, variables, and methods in monitor mixin classes should be
  // prefixed with "mm" (to avoid collision with the classes they mix in to).

  MonitorMixinBase();
  MonitorMixinBase(string& title);

  virtual void compute(vector<UInt> activeColumns, bool learn);// , string sequenceLabel = "");
  virtual void reset();

  virtual void mmClearHistory();

  virtual vector<Trace<vector<int>>> mmGetDefaultTraces(int verbosity = 1);
  virtual vector<Metric<vector<int>>> mmGetDefaultMetrics(int verbosity = 1);

  virtual string mmPrettyPrintTraces(vector<Trace<vector<int>>>& traces, Trace<vector<int>>& breakOnResets);
  virtual string mmPrettyPrintMetrics(vector<Metric<vector<int>>>& metrics);

}; // MonitorMixinBase


// Mixin for TemporalMemory that stores a detailed history, 
// for inspection and debugging.

class TemporalMemoryMonitorMixin : public MonitorMixinBase
{
protected:
  bool _mmResetActive;
  bool _mmTransitionTracesStale;

public:
  TemporalMemoryMonitorMixin() {};
  TemporalMemoryMonitorMixin(string& title);

  Trace<vector<int>>& mmGetTraceActiveColumns();
  Trace<vector<int>>& mmGetTracePredictiveCells();
  Trace<vector<int>>& mmGetTraceNumSegments();
  Trace<vector<int>>& mmGetTraceNumSynapses();
  Trace<vector<int>>& mmGetTraceSequenceLabels();
  Trace<vector<int>>& mmGetTraceResets();
  Trace<vector<int>>& mmGetTracePredictedActiveCells();
  Trace<vector<int>>& mmGetTracePredictedInactiveCells();
  Trace<vector<int>>& mmGetTracePredictedActiveColumns();
  Trace<vector<int>>& mmGetTracePredictedInactiveColumns();
  Trace<vector<int>>& mmGetTraceUnpredictedActiveColumns();

  Metric<vector<int>> mmGetMetricSequencesPredictedActiveCellsPerColumn();
  Metric<vector<int>> mmGetMetricSequencesPredictedActiveCellsShared();

  Metric<vector<int>> mmGetMetricFromTrace(Trace<vector<int>>& trace);

  string mmPrettyPrintConnections();
  string mmPrettyPrintSequenceCellRepresentations(string sortby = "Column");


  // ==============================
  // Helper methods
  // ==============================

  void _mmComputeTransitionTraces();


  //==============================
  // Overrides
  // ==============================

  virtual void compute(UInt activeColumns[], bool learn);
  virtual void reset();
  
  virtual vector<Trace<vector<int>>> mmGetDefaultTraces(int verbosity = 1);
  virtual vector<Metric<vector<int>>> mmGetDefaultMetrics(int verbosity = 1);

  virtual void mmClearHistory();

}; // of TemporalMemoryMonitorMixin

#endif // of NTA_MONITOR_MIXIN_HPP
