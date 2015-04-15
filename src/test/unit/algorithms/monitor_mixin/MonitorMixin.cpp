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

//from collections import defaultdict
//from prettytable import PrettyTable

#include "Trace.hpp"
#include "Metric.hpp"

// MonitorMixinBase class used in monitor mixin framework.
//import abc
//from prettytable import PrettyTable

class MonitorMixinBase
{
  // Base class for MonitorMixin.Each subclass will be a mixin for a particular algorithm.
  // 
  // All arguments, variables, and methods in monitor mixin classes should be
  // prefixed with "mm" (to avoid collision with the classes they mix in to).

public:
  std::string _mmName;

  //__metaclass__ = abc.ABCMeta

  MonitorMixinBase(std::string& title)
  {
    // Note : If you set the kwarg "mmName", then pretty - printing of traces and
    //        metrics will include the name you specify as a tag before every title.
    _mmName = title;

    //super(MonitorMixinBase, self).__init__(*args, **kwargs)

    // Mapping from key(string) = > trace(Trace)
    _mmTraces = None
      _mmData = None
      mmClearHistory();
  }


  void mmClearHistory()
  {
    // Clears the stored history.
    _mmTraces = {}
      _mmData = {}
  }


@staticmethod
def mmPrettyPrintTraces(traces, breakOnResets = None) :
"""
Returns pretty - printed table of traces.

@param traces(list) Traces to print in table
@param breakOnResets(BoolsTrace) Trace of resets to break table on

@return (string)Pretty - printed table of traces.
"""
assert len(traces) > 0, "No traces found"
table = PrettyTable(["#"] + [trace.prettyPrintTitle() for trace in traces])

for i in xrange(len(traces[0].data)) :
if breakOnResets and breakOnResets.data[i] :
table.add_row(["<reset>"] * (len(traces) + 1))
table.add_row([i] +
[trace.prettyPrintDatum(trace.data[i]) for trace in traces])

return table.get_string().encode("utf-8")


@staticmethod
def mmPrettyPrintMetrics(metrics) :
"""
Returns pretty - printed table of metrics.

@param metrics(list) Traces to print in table

@return (string)Pretty - printed table of metrics.
"""
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


  def mmGetDefaultTraces(self, verbosity = 1) :
  """
  Returns list of default traces. (To be overridden.)

  @param verbosity(int) Verbosity level

  @return (list)Default traces
  """
  return[]


  def mmGetDefaultMetrics(self, verbosity = 1) :
  """
  Returns list of default metrics. (To be overridden.)

  @param verbosity(int) Verbosity level

  @return (list)Default metrics
  """
  return[]

class TemporalMemoryMonitorMixin : public MonitorMixinBase
{
public:
  bool _mmResetActive;
  bool _mmTransitionTracesStale;

  std::map<std::string, Trace> _mmTraces;

  // Mixin for TemporalMemory that stores a detailed history, for inspection and debugging.

  TemporalMemoryMonitorMixin()
  {
    //MonitorMixinBase();

    _mmResetActive = true; // First iteration is always a reset
  }


  Trace& mmGetTraceActiveColumns() const
  {
    //@return (Trace) Trace of active columns
    return _mmTraces["activeColumns"];
  }


  Trace& mmGetTracePredictiveCells() const
  {
    //@return (Trace) Trace of predictive cells
    return _mmTraces["predictiveCells"];
  }


  Trace& mmGetTraceNumSegments() const
  {
    //@return (Trace) Trace of # segments
    return _mmTraces["numSegments"];
  }


  Trace& mmGetTraceNumSynapses() const
  {
    //@return (Trace) Trace of # synapses
    return _mmTraces["numSynapses"];
  }


  Trace& mmGetTraceSequenceLabels() const
  {
    //@return (Trace) Trace of sequence labels
    return _mmTraces["sequenceLabels"];
  }


  Trace& mmGetTraceResets() const
  {
    //@return (Trace) Trace of resets
    return _mmTraces["resets"];
  }


  Trace& mmGetTracePredictedActiveCells() const
  {
    //@return (Trace) Trace of predicted => active cells
    _mmComputeTransitionTraces();
    return _mmTraces["predictedActiveCells"];
  }


  Trace& mmGetTracePredictedInactiveCells() const
  {
    //@return (Trace) Trace of predicted => inactive cells
    _mmComputeTransitionTraces();
    return _mmTraces["predictedInactiveCells"];
  }


  Trace& mmGetTracePredictedActiveColumns() const
  {
    //@return (Trace) Trace of predicted => active columns
    _mmComputeTransitionTraces();
    return _mmTraces["predictedActiveColumns"];
  }


  Trace& mmGetTracePredictedInactiveColumns() const
  {
    //@return (Trace) Trace of predicted => inactive columns
    _mmComputeTransitionTraces();
    return _mmTraces["predictedInactiveColumns"];
  }


  Trace& mmGetTraceUnpredictedActiveColumns() const
  {
    //@return (Trace) Trace of unpredicted => active columns
    _mmComputeTransitionTraces();
    return _mmTraces["unpredictedActiveColumns"];
  }


  Metric& mmGetMetricFromTrace(Trace& trace) const
  {
    // Convenience method to compute a metric over 
    // an indices trace, excluding resets.
    //@param (IndicesTrace) Trace of indices
    //@return (Metric) Metric over trace excluding resets
    return Metric.createFromTrace(trace.makeCountsTrace(),
      mmGetTraceResets());
  }


  Metric& mmGetMetricSequencesPredictedActiveCellsPerColumn() const
  {
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
  }

  Metric& mmGetMetricSequencesPredictedActiveCellsShared() const
  {
    """
      Metric for number of sequences each predicted = > active cell appears in

      Note : This metric is flawed when it comes to high - order sequences.

             @return (Metric)metric
             """
             _mmComputeTransitionTraces()

             numSequencesForCell = defaultdict(lambda: 0)

             for predictedActiveCells in(
               _mmData["predictedActiveCellsForSequence"].values()) :
               for cell in predictedActiveCells :
    numSequencesForCell[cell] += 1

      return Metric(self,
      "# sequences each predicted => active cells appears in",
      numSequencesForCell.values())
  }


  std::string mmPrettyPrintConnections()
  {
    """
      Pretty print the connections in the temporal memory.

      TODO: Use PrettyTable.

            @return (string)Pretty - printed text
            """
            text = ""

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

            return text
  }


  std::string mmPrettyPrintSequenceCellRepresentations(self, sortby="Column"):
  {
    """
      Pretty print the cell representations for sequences in the history.

      @param sortby(string) Column of table to sort by

      @return (string)Pretty - printed text
      """
      _mmComputeTransitionTraces()
      table = PrettyTable(["Pattern", "Column", "predicted=>active cells"])

      for sequenceLabel, predictedActiveCells in(
        _mmData["predictedActiveCellsForSequence"].iteritems()) :
        cellsForColumn = mapCellsToColumns(predictedActiveCells)
        for column, cells in cellsForColumn.iteritems() :
          table.add_row([sequenceLabel, column, list(cells)])

          return table.get_string(sortby = sortby).encode("utf-8")
  }


  // ==============================
  // Helper methods
  // ==============================

  void _mmComputeTransitionTraces()
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

    _mmData["predictedActiveCellsForSequence"] = defaultdict(set)

    _mmTraces["predictedActiveCells"] = IndicesTrace(self,
    "predicted => active cells (correct)")
    _mmTraces["predictedInactiveCells"] = IndicesTrace(self,
    "predicted => inactive cells (extra)")
    _mmTraces["predictedActiveColumns"] = IndicesTrace(self,
    "predicted => active columns (correct)")
    _mmTraces["predictedInactiveColumns"] = IndicesTrace(self,
    "predicted => inactive columns (extra)")
    _mmTraces["unpredictedActiveColumns"] = IndicesTrace(self,
    "unpredicted => active columns (bursting)")

    predictedCellsTrace = _mmTraces["predictedCells"]

      for i, activeColumns in enumerate(mmGetTraceActiveColumns().data) :
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

      _mmTransitionTracesStale = false
  }

  # ==============================
  # Overrides
  # ==============================

    void compute(self, activeColumns, sequenceLabel = None, **kwargs)
  {
    _mmTraces["predictedCells"].data.append(predictiveCells)

      super(TemporalMemoryMonitorMixin, self).compute(activeColumns, **kwargs)

      _mmTraces["predictiveCells"].data.append(predictiveCells)
      _mmTraces["activeColumns"].data.append(activeColumns)

      _mmTraces["numSegments"].data.append(connections.numSegments())
      _mmTraces["numSynapses"].data.append(connections.numSynapses())

      _mmTraces["sequenceLabels"].data.append(sequenceLabel)
      _mmTraces["resets"].data.append(_mmResetActive)
      _mmResetActive = False

      _mmTransitionTracesStale = True
  }


  void reset()
  {
    super(TemporalMemoryMonitorMixin, self).reset()

      _mmResetActive = True
  }


  Trace mmGetDefaultTraces(self, verbosity=1)
  {
    traces = [
      mmGetTraceActiveColumns(),
        mmGetTracePredictedActiveColumns(),
        mmGetTracePredictedInactiveColumns(),
        mmGetTraceUnpredictedActiveColumns(),
        mmGetTracePredictedActiveCells(),
        mmGetTracePredictedInactiveCells()
    ]

    if verbosity == 1:
      traces = [trace.makeCountsTrace() for trace in traces]

        traces += [
          mmGetTraceNumSegments(),
            mmGetTraceNumSynapses()
        ]

        return traces + [mmGetTraceSequenceLabels()]
  }


  Metric mmGetDefaultMetrics(self, verbosity = 1)
  {
    resetsTrace = mmGetTraceResets()
      return ([Metric.createFromTrace(trace, excludeResets = resetsTrace)
      for trace in mmGetDefaultTraces()[:-3]] +
        [Metric.createFromTrace(trace)
        for trace in mmGetDefaultTraces()[-3:-1]] +
          [mmGetMetricSequencesPredictedActiveCellsPerColumn(),
          mmGetMetricSequencesPredictedActiveCellsShared()])
  }


  void mmClearHistory()
  {
    //super(TemporalMemoryMonitorMixin, self).mmClearHistory()

    _mmTraces["predictedCells"] = IndicesTrace("predicted cells");
    _mmTraces["activeColumns"] = IndicesTrace("active columns");
    _mmTraces["predictiveCells"] = IndicesTrace("predictive cells");
    _mmTraces["numSegments"] = CountsTrace("# segments");
    _mmTraces["numSynapses"] = CountsTrace("# synapses");
    _mmTraces["sequenceLabels"] = StringsTrace("sequence labels");
    _mmTraces["resets"] = BoolsTrace("resets");

    _mmTransitionTracesStale = true;
  }
