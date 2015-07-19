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
* Implementation of MonitorMixinBase class used in 
* monitor mixin framework.
*/

#include <nupic/utils/PatternMachine.hpp>
#include "MonitorMixin.hpp"

MonitorMixinBase::MonitorMixinBase()
{
  _mmName = "";
}

MonitorMixinBase::MonitorMixinBase(string& title)
{
  // Note : If you set the kwarg "mmName", then pretty - printing of traces and
  //        metrics will include the name you specify as a tag before every title.
  _mmName = title;

  // Mapping from key(string) = > trace(Trace)
  _mmTraces_UInt.clear();
  _mmTraces_string.clear();
  _mmTraces_bool.clear();

  _mmData.clear();

  mmClearHistory();
}

void MonitorMixinBase::mmReset()
{
  _mmName = "";
  
  mmClearHistory();
}

void MonitorMixinBase::mmClearHistory()
{
  // Clears the stored history.
  _mmTraces_UInt.clear();
  _mmTraces_string.clear();
  _mmTraces_bool.clear();

  _mmData.clear();
}

string MonitorMixinBase::mmPrettyPrintTraces(vector<Trace<vector<UInt>>>& traces, Trace<vector<bool>>& breakOnResets)
{
  //Returns pretty - printed table of traces.
  //@param traces(list) Traces to print in table
  //@param breakOnResets(BoolsTrace) Trace of resets to break table on
  //@return (string)Pretty - printed table of traces.
  /*
  NTA_ASSERT(traces.size() > 0);
  table = PrettyTable(["#"] + [trace.prettyPrintTitle() for trace in traces])

    for i in xrange(len(traces[0].data)) :
      if breakOnResets and breakOnResets.data[i] :
        table.add_row(["<reset>"] * (len(traces) + 1))
        table.add_row([i] +
        [trace.prettyPrintDatum(trace.data[i]) for trace in traces])

        return table.get_string().encode("utf-8")
  */
  return "";
}

string MonitorMixinBase::mmPrettyPrintMetrics(vector<MetricsVector>& metrics)
{
  //Returns pretty - printed table of metrics.
  //@param metrics(list) Traces to print in table
  //@return (string)Pretty - printed table of metrics.
  /*
  assert len(metrics) > 0, "No metrics found"
    table = PrettyTable(["Metric",
    "min", "max", "sum", "mean", "standard deviation"])

    for metric in metrics :
  table.add_row([metric.prettyPrintTitle(),
    metric.min,
    metric.max,
    metric.sum,
    metric.mean,
    metric.standardDeviation])

    return table.get_string().encode("utf-8")
  */
  return "";
}

string TemporalMemoryMonitorMixin::mmPrettyPrintConnections()
{
  //Pretty print the connections in the temporal memory.
  //TODO: Use PrettyTable.
  //@return (string)Pretty - printed text
  string text = "";

  text += "(numSegments) [(source cell=permanence ...), ...]\n";
  text += "------------------------------------\n";

  vector<UInt> columns = nupic::utils::range(numberOfColumns());

  for (UInt column : columns)
  {
    vector<Cell> cells = cellsForColumn(column);

    for (Cell cell : cells)
    {
      vector<Segment> segmentList;

      text += string("Column ") + ::to_string(column) + string(" / Cell ") + ::to_string(cell.idx);

      segmentList = connections.segmentsForCell(cell);
      text += string(":\t(") + ::to_string(segmentList.size()) + string(") [");

      for (Segment seg : segmentList)
      {
        text += string("(");

        vector<SynapseData> synapseList;
        for (Synapse synapse : connections.synapsesForSegment(seg))
        {
          SynapseData data = connections.dataForSynapse(synapse);
          synapseList.push_back(data);
        }
        //sort(synapseList.begin(), synapseList.end());

        string synapseStringList = "";
        for (SynapseData synapseData : synapseList)
        {
          synapseStringList += ::to_string(synapseData.presynapticCell.idx) + "=" + ::to_string(synapseData.permanence);
          if (synapseList.size() > 1)
            synapseStringList += " ";
        }
        text += synapseStringList + ")";
        if (segmentList.size() > 1)
          text += ",";
      }
      text += "]\n";
    }

    if (column < columns.size() - 1)  // not last
      text += "\n";
  }

  text += "------------------------------------\n";

  return text;
}

string TemporalMemoryMonitorMixin::mmPrettyPrintSequenceCellRepresentations(string sortby)
{
  //Pretty print the cell representations for sequences in the history.
  //@param sortby(string) Column of table to sort by
  //@return (string)Pretty - printed text
  /*
  _mmComputeTransitionTraces()
  table = PrettyTable(["Pattern", "Column", "predicted=>active cells"])

  for sequenceLabel, predictedActiveCells in(
  _mmData["predictedActiveCellsForSequence"].iteritems()) :
  cellsForColumn = mapCellsToColumns(predictedActiveCells)
  for column, cells in cellsForColumn.iteritems() :
  table.add_row([sequenceLabel, column, list(cells)])

  return table.get_string(sortby = sortby).encode("utf-8")
  */
  return "";
}

vector<Trace<vector<UInt>>> MonitorMixinBase::mmGetDefaultTraces(int verbosity)
{
  //Returns list of default traces. (To be overridden.)
  //@param verbosity(int) Verbosity level
  //@return (list)Default traces
  return {};
}

vector<MetricsVector> MonitorMixinBase::mmGetDefaultMetrics(int verbosity)
{
  //Returns list of default metrics. (To be overridden.)
  //@param verbosity(int) Verbosity level
  //@return (list)Default metrics
  return {};
}

TemporalMemoryMonitorMixin::TemporalMemoryMonitorMixin(string& title)
{
  _mmName = title;
  _mmResetActive = true; // First iteration is always a reset
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTraceActiveColumns()
{
  //@return (Trace) Trace of active columns
  return _mmTraces_UInt.find("activeColumns")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTracePredictiveCells()
{
  //@return (Trace) Trace of predictive cells
  return _mmTraces_UInt.find("predictiveCells")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTraceNumSegments()
{
  //@return (Trace) Trace of # segments
  return _mmTraces_UInt.find("numSegments")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTraceNumSynapses()
{
  //@return (Trace) Trace of # synapses
  return _mmTraces_UInt.find("numSynapses")->second;
}

Trace<vector<string>>& TemporalMemoryMonitorMixin::mmGetTraceSequenceLabels()
{
  //@return (Trace) Trace of sequence labels
  return _mmTraces_string.find("sequenceLabels")->second;
}

Trace<vector<bool>>& TemporalMemoryMonitorMixin::mmGetTraceResets()
{
  //@return (Trace) Trace of resets
  return _mmTraces_bool.find("resets")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTracePredictedActiveCells()
{
  //@return (Trace) Trace of predicted => active cells
  _mmComputeTransitionTraces();
  return _mmTraces_UInt.find("predictedActiveCells")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTracePredictedInactiveCells()
{
  //@return (Trace) Trace of predicted => inactive cells
  _mmComputeTransitionTraces();
  return _mmTraces_UInt.find("predictedInactiveCells")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTracePredictedActiveColumns()
{
  //@return (Trace) Trace of predicted => active columns
  _mmComputeTransitionTraces();
  return _mmTraces_UInt.find("predictedActiveColumns")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTracePredictedInactiveColumns()
{
  //@return (Trace) Trace of predicted => inactive columns
  _mmComputeTransitionTraces();
  return _mmTraces_UInt.find("predictedInactiveColumns")->second;
}

Trace<vector<UInt>>& TemporalMemoryMonitorMixin::mmGetTraceUnpredictedActiveColumns()
{
  //@return (Trace) Trace of unpredicted => active columns
  _mmComputeTransitionTraces();
  return _mmTraces_UInt.find("unpredictedActiveColumns")->second;
}

MetricsVector TemporalMemoryMonitorMixin::mmGetMetricSequencesPredictedActiveCellsPerColumn()
{
  // Metric for number of predicted => active cells per column for each sequence
  //@return (Metric) metric
  _mmComputeTransitionTraces();

  vector<UInt> numCellsPerColumn;
/*
  for (Cell predictedActiveCells : _mmData["predictedActiveCellsForSequence"])
  {
    map<Int, vector<Cell>> cellsForColumn = mapCellsToColumns(predictedActiveCells);
    numCellsPerColumn += [len(x) for x in cellsForColumn.values()];
  }
*/
  MetricsVector ret;//(*this, string("temp"), numCellsPerColumn);
  return ret;
}


MetricsVector TemporalMemoryMonitorMixin::mmGetMetricSequencesPredictedActiveCellsShared()
{
  //Metric for number of sequences each predicted = > active cell appears in
  //Note : This metric is flawed when it comes to high - order sequences.
  //@return (Metric)metric
  _mmComputeTransitionTraces();

  map<Cell, int> numSequencesForCell;

  for (Cell predictedActiveCells : _mmData["predictedActiveCellsForSequence"])
  {
    numSequencesForCell[predictedActiveCells] += 1;
  }

  vector<UInt> numCellsPerColumn;
  MetricsVector ret;// (*this, string("# sequences each predicted => active cells appears in"), numCellsPerColumn);// numSequencesForCell);
  return ret;
}

// ==============================
// Helper methods
// ==============================

void TemporalMemoryMonitorMixin::_mmComputeTransitionTraces()
{
  // Computes the transition traces, if necessary.
  //
  // Transition traces are the following :

  // predicted = > active cells
  // predicted = > inactive cells
  // predicted = > active columns
  // predicted = > inactive columns
  // unpredicted = > active columns
  //
  if (!_mmTransitionTracesStale)
    return;

  _mmData["predictedActiveCellsForSequence"] = vector<Cell>();

  _mmTraces_UInt["predictedActiveCells"] = IndicesTrace(this, string("predicted => active cells (correct)"));
  _mmTraces_UInt["predictedInactiveCells"] = IndicesTrace(this, string("predicted => inactive cells (extra)"));
  _mmTraces_UInt["predictedActiveColumns"] = IndicesTrace(this, string("predicted => active columns (correct)"));
  _mmTraces_UInt["predictedInactiveColumns"] = IndicesTrace(this, string("predicted => inactive columns (extra)"));
  _mmTraces_UInt["unpredictedActiveColumns"] = IndicesTrace(this, string("unpredicted => active columns (bursting)"));

/*
  Trace<vector<UInt>>& predictedCellsTrace = _mmTraces_UInt.find("predictedCells")->second;

  set<Cell> predictedActiveCells;
  set<Cell> predictedInactiveCells;
  set<UInt> predictedActiveColumns;
  set<UInt> predictedInactiveColumns;

  UInt i = 0;
  for (UInt activeColumns : mmGetTraceActiveColumns()._data[i])
  {
    for (Cell predictedCell : predictedCellsTrace._data[i])
    {
      UInt predictedColumn = columnForCell(predictedCell);

      if (activeColumns.find(activeColumns.begin, activeColumns.end(), predictedColumn) != activeColumns.end())
      {
        predictedActiveCells.insert(predictedCell);
        predictedActiveColumns.insert(predictedColumn);

        string sequenceLabel = mmGetTraceSequenceLabels()._data;
        if (sequenceLabel.size() > 0)
          _mmData["predictedActiveCellsForSequence"][sequenceLabel].add(predictedCell);
      }
      else
      {
        predictedInactiveCells.insert(predictedCell);
        predictedInactiveColumns.insert(predictedColumn);
      }
    }
    i++;
  }
  unpredictedActiveColumns = activeColumns - predictedActiveColumns;

  _mmTraces_UInt["predictedActiveCells"]._data.push_back(predictedActiveCells);
  _mmTraces_UInt["predictedInactiveCells"]._data.push_back(predictedInactiveCells);
  _mmTraces_UInt["predictedActiveColumns"]._data.push_back(predictedActiveColumns);
  _mmTraces_UInt["predictedInactiveColumns"]._data.push_back(predictedInactiveColumns);
  _mmTraces_UInt["unpredictedActiveColumns"]._data.push_back(unpredictedActiveColumns);
*/
  _mmTransitionTracesStale = false;
}

//==============================
// Overrides
// ==============================

void TemporalMemoryMonitorMixin::mmCompute(UInt activeColumnsSize, UInt activeColumns[], bool learn, string sequenceLabel)
{
//  _mmTraces_UInt["predictedCells"]._data.push_back(predictiveCells);

  compute(activeColumnsSize, &activeColumns[0], learn);
/*
  _mmTraces_UInt["predictiveCells"]._data.push_back(predictiveCells);
  _mmTraces_UInt["activeColumns"]._data.push_back(activeColumns);

  _mmTraces_UInt["numSegments"]._data.push_back(connections.numSegments());
  _mmTraces_UInt["numSynapses"]._data.push_back(connections.numSynapses());

  _mmTraces_string["sequenceLabels"]._data.push_back({ sequenceLabel });
*/
  _mmTraces_bool["resets"]._data.push_back({ _mmResetActive });
  _mmResetActive = false;

  _mmTransitionTracesStale = true;
}

void TemporalMemoryMonitorMixin::mmReset()
{
  TemporalMemory::reset();
  //MonitorMixinBase::reset();
  _mmResetActive = true;
}

vector<Trace<vector<UInt>>> TemporalMemoryMonitorMixin::mmGetDefaultTraces(int verbosity)
{
  vector<Trace<vector<UInt>>> traces;
    
  traces.push_back(mmGetTraceActiveColumns());
  traces.push_back(mmGetTracePredictedActiveColumns());
  traces.push_back(mmGetTracePredictedInactiveColumns());
  traces.push_back(mmGetTraceUnpredictedActiveColumns());
  traces.push_back(mmGetTracePredictedActiveCells());
  traces.push_back(mmGetTracePredictedInactiveCells());

  if (verbosity == 1)
  {
    for (int i = 0; i < 5; i++)
    {
      Trace<vector<UInt>>& trace = traces[i];
      trace.makeCountsTrace();
    }

    traces.push_back(mmGetTraceNumSegments());
    traces.push_back(mmGetTraceNumSynapses());
  }
//  traces.push_back(mmGetTraceSequenceLabels());

  return traces;
}

MetricsVector TemporalMemoryMonitorMixin::mmGetMetricFromTrace(Trace<vector<UInt>>& trace)
{
  // Convenience method to compute a metric over 
  // an indices trace, excluding resets.
  //@param (IndicesTrace) Trace of indices
  //@return (Metric) Metric over trace excluding resets
  Trace<vector<UInt>> countsTrace(trace.makeCountsTrace());
  MetricsVector ret;
  ret.createFromTrace(countsTrace, mmGetTraceResets());
  return ret;
}

vector<MetricsVector> TemporalMemoryMonitorMixin::mmGetDefaultMetrics(int verbosity)
{
  vector<MetricsVector> ret;
  return ret;

//  ([Metric.createFromTrace(trace, excludeResets = resetsTrace)
//    for trace in mmGetDefaultTraces()[:-3]] +
//      [Metric.createFromTrace(trace)
//      for trace in mmGetDefaultTraces()[-3:-1]] +
//        [mmGetMetricSequencesPredictedActiveCellsPerColumn(),
//        mmGetMetricSequencesPredictedActiveCellsShared()])
}

void TemporalMemoryMonitorMixin::mmClearHistory()
{
  MonitorMixinBase::mmClearHistory();

  _mmTraces_UInt["predictedCells"] = IndicesTrace(this, string("predicted cells"));
  _mmTraces_UInt["activeColumns"] = IndicesTrace(this, string("active columns"));
  _mmTraces_UInt["predictiveCells"] = IndicesTrace(this, string("predictive cells"));
  _mmTraces_UInt["numSegments"] = CountsTrace(this, string("# segments"));
  _mmTraces_UInt["numSynapses"] = CountsTrace(this, string("# synapses"));
  _mmTraces_string["sequenceLabels"] = StringsTrace(this, string("sequence labels"));
  _mmTraces_bool["resets"] = BoolsTrace(this, string("resets"));

  _mmTransitionTracesStale = true;
}
