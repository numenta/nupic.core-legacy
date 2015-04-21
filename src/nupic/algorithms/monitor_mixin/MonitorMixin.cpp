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
  _mmTraces.clear();
  //_mmData.clear();
  mmClearHistory();
}


void MonitorMixinBase::reset()
{
  _mmName = "";
  
  _mmTraces.clear();
  //_mmData.clear();
  
  mmClearHistory();
}


void MonitorMixinBase::compute(vector<UInt> activeColumns, bool learn)
{
}


void MonitorMixinBase::mmClearHistory()
{
  // Clears the stored history.
  _mmTraces.clear();
  //_mmData.clear();
}


string MonitorMixinBase::mmPrettyPrintTraces(vector<Trace<vector<int>>>& traces, Trace<vector<int>>& breakOnResets)
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

string MonitorMixinBase::mmPrettyPrintMetrics(vector<Metric<vector<int>>>& metrics)
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


vector<Trace<vector<int>>> MonitorMixinBase::mmGetDefaultTraces(int verbosity)
{
  //Returns list of default traces. (To be overridden.)
  //@param verbosity(int) Verbosity level
  //@return (list)Default traces
  return {};
}


vector<Metric<vector<int>>> MonitorMixinBase::mmGetDefaultMetrics(int verbosity)
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


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceActiveColumns()
{
  //@return (Trace) Trace of active columns
  return _mmTraces.find("activeColumns")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTracePredictiveCells()
{
  //@return (Trace) Trace of predictive cells
  return _mmTraces.find("predictiveCells")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceNumSegments()
{
  //@return (Trace) Trace of # segments
  return _mmTraces.find("numSegments")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceNumSynapses()
{
  //@return (Trace) Trace of # synapses
  return _mmTraces.find("numSynapses")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceSequenceLabels()
{
  //@return (Trace) Trace of sequence labels
  return _mmTraces.find("sequenceLabels")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceResets()
{
  //@return (Trace) Trace of resets
  return _mmTraces.find("resets")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTracePredictedActiveCells()
{
  //@return (Trace) Trace of predicted => active cells
  _mmComputeTransitionTraces();
  return _mmTraces.find("predictedActiveCells")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTracePredictedInactiveCells()
{
  //@return (Trace) Trace of predicted => inactive cells
  _mmComputeTransitionTraces();
  return _mmTraces.find("predictedInactiveCells")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTracePredictedActiveColumns()
{
  //@return (Trace) Trace of predicted => active columns
  _mmComputeTransitionTraces();
  return _mmTraces.find("predictedActiveColumns")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTracePredictedInactiveColumns()
{
  //@return (Trace) Trace of predicted => inactive columns
  _mmComputeTransitionTraces();
  return _mmTraces.find("predictedInactiveColumns")->second;
}


Trace<vector<int>>& TemporalMemoryMonitorMixin::mmGetTraceUnpredictedActiveColumns()
{
  //@return (Trace) Trace of unpredicted => active columns
  _mmComputeTransitionTraces();
  return _mmTraces.find("unpredictedActiveColumns")->second;
}


Metric<vector<int>> TemporalMemoryMonitorMixin::mmGetMetricFromTrace(Trace<vector<int>>& trace)
{
  // Convenience method to compute a metric over 
  // an indices trace, excluding resets.
  //@param (IndicesTrace) Trace of indices
  //@return (Metric) Metric over trace excluding resets
  Trace<vector<int>> countsTrace = trace.makeCountsTrace();
  return Metric<vector<int>>();// Metric<vector<int>>::createFromTrace(countsTrace, false);//, mmGetTraceResets());
}


Metric<vector<int>> TemporalMemoryMonitorMixin::mmGetMetricSequencesPredictedActiveCellsPerColumn()
{
/*
  vector<int> data;
  Instance temp;
  Metric<vector<int>> ret(this, string("temp"), &data);

  // Metric for number of predicted => active cells per column for each sequence
  //@return (Metric) metric
  _mmComputeTransitionTraces();

  numCellsPerColumn = []

  for predictedActiveCells in(
  _mmData["predictedActiveCellsForSequence"].values()) :
  cellsForColumn = mapCellsToColumns(predictedActiveCells)
  numCellsPerColumn += [len(x) for x in cellsForColumn.values()]

  return Metric(self,
    "# predicted => active cells per column for each sequence",
    numCellsPerColumn)
*/
  return Metric<vector<int>>();
}


Metric<vector<int>> TemporalMemoryMonitorMixin::mmGetMetricSequencesPredictedActiveCellsShared()
{
/*
  //Metric for number of sequences each predicted = > active cell appears in
  //Note : This metric is flawed when it comes to high - order sequences.
  //@return (Metric)metric
  _mmComputeTransitionTraces();

  numSequencesForCell = defaultdict(lambda: 0);

  for predictedActiveCells in(
  _mmData["predictedActiveCellsForSequence"].values()) :
  for cell in predictedActiveCells :
  numSequencesForCell[cell] += 1;

  return Metric(self,
    "# sequences each predicted => active cells appears in",
    numSequencesForCell.values());
*/
  return Metric<vector<int>>();
}


string TemporalMemoryMonitorMixin::mmPrettyPrintConnections()
{
  //Pretty print the connections in the temporal memory.
  //TODO: Use PrettyTable.
  //@return (string)Pretty - printed text
  string text = "";

  /*
  text += ("Segments: (format => "
  "(#) [(source cell=permanence ...),       ...]\n")
  text += "------------------------------------\n"

  columns = range(numberOfColumns())

  for column in columns :
  cells = cellsForColumn(column)

  for cell in cells :
  segmentDict = dict()

  for seg in connections.segmentsForCell(cell) :
  synapseList = []

  for synapse in connections.synapsesForSegment(seg) :
  (_, sourceCell, permanence) = connections.dataForSynapse(
  synapse)

  synapseList.append((sourceCell, permanence))

  synapseList.sort()
  synapseStringList = ["{0:3}={1:.2f}".format(sourceCell, permanence) for
  sourceCell, permanence in synapseList]
  segmentDict[seg] = "({0})".format(" ".join(synapseStringList))

  text += ("Column {0:3} / Cell {1:3}:\t({2}) {3}\n".format(
  column, cell,
  len(segmentDict.values()),
  "[{0}]".format(",       ".join(segmentDict.values()))))

  if column < len(columns) - 1:  # not last
  text += "\n"

  text += "------------------------------------\n"
  */
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

  //_mmData["predictedActiveCellsForSequence"] = defaultdict(set);

//    _mmTraces["predictedActiveCells"] = IndicesTrace("predicted => active cells (correct)");
//    _mmTraces["predictedInactiveCells"] = IndicesTrace("predicted => inactive cells (extra)");
//    _mmTraces["predictedActiveColumns"] = IndicesTrace("predicted => active columns (correct)");
//    _mmTraces["predictedInactiveColumns"] = IndicesTrace("predicted => inactive columns (extra)");
//    _mmTraces["unpredictedActiveColumns"] = IndicesTrace("unpredicted => active columns (bursting)");

  Trace<vector<int>>& predictedCellsTrace = _mmTraces.find("predictedCells")->second;

  /*      for i, activeColumns in enumerate(mmGetTraceActiveColumns().data) :
          predictedActiveCells = set()
          predictedInactiveCells = set()
          predictedActiveColumns = set()
          predictedInactiveColumns = set()

          for predictedCell in predictedCellsTrace.data[i]:
      predictedColumn = columnForCell(predictedCell)

        if predictedColumn  in activeColumns :
      predictedActiveCells.add(predictedCell)
        predictedActiveColumns.add(predictedColumn)

        sequenceLabel = mmGetTraceSequenceLabels().data[i]
        if sequenceLabel is not None :
          _mmData["predictedActiveCellsForSequence"][sequenceLabel].add(
          predictedCell)
        else:
      predictedInactiveCells.add(predictedCell)
        predictedInactiveColumns.add(predictedColumn)

        unpredictedActiveColumns = activeColumns - predictedActiveColumns

        _mmTraces["predictedActiveCells"].data.append(predictedActiveCells)
        _mmTraces["predictedInactiveCells"].data.append(predictedInactiveCells)
        _mmTraces["predictedActiveColumns"].data.append(predictedActiveColumns)
        _mmTraces["predictedInactiveColumns"].data.append(predictedInactiveColumns)
        _mmTraces["unpredictedActiveColumns"].data.append(unpredictedActiveColumns)
    */
  _mmTransitionTracesStale = false;
}


//==============================
// Overrides
// ==============================

void TemporalMemoryMonitorMixin::compute(UInt activeColumns[], bool learn)
{
  //_mmTraces["predictedCells"].data.append(predictiveCells)

  //super(TemporalMemoryMonitorMixin, self).compute(activeColumns, **kwargs)

  //_mmTraces["predictiveCells"].data.append(predictiveCells)
  //_mmTraces["activeColumns"].data.append(activeColumns)

  //_mmTraces["numSegments"].data.append(connections.numSegments())
  //_mmTraces["numSynapses"].data.append(connections.numSynapses())

  //_mmTraces["sequenceLabels"].data.append(sequenceLabel)
  //_mmTraces["resets"].data.append(_mmResetActive)
  _mmResetActive = false;

  _mmTransitionTracesStale = true;
}


void TemporalMemoryMonitorMixin::reset()
{
  //MonitorMixinBase::reset();
  _mmResetActive = true;
}


vector<Trace<vector<int>>> TemporalMemoryMonitorMixin::mmGetDefaultTraces(int verbosity)
{
  vector<Trace<vector<int>>> traces;
    
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
      Trace<vector<int>>& trace = traces[i];
      trace.makeCountsTrace();
    }

    traces.push_back(mmGetTraceNumSegments());
    traces.push_back(mmGetTraceNumSynapses());
  }
  traces.push_back(mmGetTraceSequenceLabels());

  return traces;
}


vector<Metric<vector<int>>> TemporalMemoryMonitorMixin::mmGetDefaultMetrics(int verbosity)
{
  vector<Metric<vector<int>>> ret;

//  ([Metric.createFromTrace(trace, excludeResets = resetsTrace)
//    for trace in mmGetDefaultTraces()[:-3]] +
//      [Metric.createFromTrace(trace)
//      for trace in mmGetDefaultTraces()[-3:-1]] +
//        [mmGetMetricSequencesPredictedActiveCellsPerColumn(),
//        mmGetMetricSequencesPredictedActiveCellsShared()])

  return ret;
}


void TemporalMemoryMonitorMixin::mmClearHistory()
{
  //super(TemporalMemoryMonitorMixin, self).mmClearHistory()

//    _mmTraces["predictedCells"] = IndicesTrace("predicted cells");
//    _mmTraces["activeColumns"] = IndicesTrace("active columns");
//    _mmTraces["predictiveCells"] = IndicesTrace("predictive cells");
//    _mmTraces["numSegments"] = CountsTrace("# segments");
//    _mmTraces["numSynapses"] = CountsTrace("# synapses");
//    _mmTraces["sequenceLabels"] = StringsTrace("sequence labels");
//    _mmTraces["resets"] = BoolsTrace("resets");

  _mmTransitionTracesStale = true;
}
