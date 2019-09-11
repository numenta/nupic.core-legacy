# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
ColumnPooler mixin that enables detailed monitoring of history.
"""

import copy

from .metric import Metric
from .monitor_mixin_base import MonitorMixinBase
from .trace import IndicesTrace, CountsTrace,BoolsTrace, StringsTrace


class ColumnPoolerMonitorMixin(MonitorMixinBase):
    """
    Mixin for ColumnPooler that stores a detailed history, for inspection and
    debugging.
    """

    def __init__(self, *args, **kwargs):
        super(ColumnPoolerMonitorMixin, self).__init__(*args, **kwargs)

        self._mmResetActive = True    # First iteration is always a reset


    def mmGetTraceActiveCells(self):
        """
        @return (Trace) Trace of active cells
        """
        return self._mmTraces["activeCells"]


    def mmGetTraceNumDistalSegments(self):
        """
        @return (Trace) Trace of # segments
        """
        return self._mmTraces["numDistalSegments"]


    def mmGetTraceNumDistalSynapses(self):
        """
        @return (Trace) Trace of # distal synapses with permanence > 0
        """
        return self._mmTraces["numDistalSynapses"]


    def mmGetTraceNumConnectedDistalSynapses(self):
        """
        @return (Trace) Trace of # connected distal synapses
        """
        return self._mmTraces["numConnectedDistalSynapses"]


    def mmGetTraceNumProximalSynapses(self):
        """
        @return (Trace) Trace of # proximal synapses with permanence > 0
        """
        return self._mmTraces["numProximalSynapses"]


    def mmGetTraceNumConnectedProximalSynapses(self):
        """
        @return (Trace) Trace of # connected proximal synapses
        """
        return self._mmTraces["numConnectedProximalSynapses"]


    def mmGetTraceSequenceLabels(self):
        """
        @return (Trace) Trace of sequence labels
        """
        return self._mmTraces["sequenceLabels"]


    def mmGetTraceResets(self):
        """
        @return (Trace) Trace of resets
        """
        return self._mmTraces["resets"]


    def mmGetMetricFromTrace(self, trace):
        """
        Convenience method to compute a metric over an indices trace, excluding
        resets.

        @param (IndicesTrace) Trace of indices

        @return (Metric) Metric over trace excluding resets
        """
        return Metric.createFromTrace(trace.makeCountsTrace(), excludeResets=self.mmGetTraceResets())


    # ==============================
    # Overrides
    # ==============================
    def compute(self,
                feedforwardInput=(),
                lateralInputs=(),
                feedforwardGrowthCandidates=None,
                learn=True,
                sequenceLabel=None,
                **kwargs):
        super(ColumnPoolerMonitorMixin, self).compute(feedforwardInput, lateralInputs, feedforwardGrowthCandidates, learn)

        self._mmTraces["activeCells"].data.append(set(self.getActiveCells()))

        self._mmTraces["numDistalSegments"].data.append(self.numberOfDistalSegments())
        self._mmTraces["numDistalSynapses"].data.append(self.numberOfDistalSynapses())
        self._mmTraces["numConnectedDistalSynapses"].data.append(self.numberOfConnectedDistalSynapses())
        self._mmTraces["numProximalSynapses"].data.append(self.numberOfProximalSynapses())
        self._mmTraces["numConnectedProximalSynapses"].data.append(self.numberOfConnectedProximalSynapses())
        self._mmTraces["sequenceLabels"].data.append(sequenceLabel)
        self._mmTraces["resets"].data.append(self._mmResetActive)
        self._mmResetActive = False

        self._mmTransitionTracesStale = True


    def reset(self):
        super(ColumnPoolerMonitorMixin, self).reset()

        self._mmResetActive = True


    def mmGetDefaultTraces(self, verbosity=1):
        traces = [self.mmGetTraceActiveCells()]

        if verbosity == 1:
            traces = [trace.makeCountsTrace() for trace in traces]

        traces += [
            self.mmGetTraceNumDistalSegments(),
            self.mmGetTraceNumDistalSynapses(),
            self.mmGetTraceNumConnectedDistalSynapses(),
            self.mmGetTraceNumProximalSynapses(),
            self.mmGetTraceNumConnectedProximalSynapses(),
            ]

        return traces + [self.mmGetTraceSequenceLabels()]


    def mmGetDefaultMetrics(self, verbosity=1):
        resetsTrace = self.mmGetTraceResets()
        return ([Metric.createFromTrace(trace, excludeResets=resetsTrace)
                    for trace in self.mmGetDefaultTraces()[:-3]] + [Metric.createFromTrace(trace)
                        for trace in self.mmGetDefaultTraces()[-3:-1]])


    def mmClearHistory(self):
        super(ColumnPoolerMonitorMixin, self).mmClearHistory()

        self._mmTraces["activeCells"] = IndicesTrace(self, "active cells")
        self._mmTraces["numDistalSegments"] = CountsTrace(self, "# distal segments")
        self._mmTraces["numDistalSynapses"] = CountsTrace(self, "# distal synapses")
        self._mmTraces["numConnectedDistalSynapses"] = CountsTrace(self, "# connected distal synapses")
        self._mmTraces["numProximalSynapses"] = CountsTrace(self, "# proximal synapses")
        self._mmTraces["numConnectedProximalSynapses"] = CountsTrace(self, "# connected proximal synapses")
        self._mmTraces["sequenceLabels"] = StringsTrace(self, "sequence labels")
        self._mmTraces["resets"] = BoolsTrace(self, "resets")
        self._mmTransitionTracesStale = True


    def mmGetCellActivityPlot(self, title="", showReset=False, resetShading=0.25, activityType="activeCells"):
        """
        Returns plot of the cell activity.
    
        @param title (string) an optional title for the figure
    
        @param showReset (bool) if true, the first set of cell activities
                                after a reset will have a gray background
    
        @param resetShading (float) if showReset is true, this float specifies the
                                    intensity of the reset background with 0.0
                                    being white and 1.0 being black
    
        @param activityType (string) The type of cell activity to display. Valid
                                     types include "activeCells"
    
        @return (Plot) plot
        """

        cellTrace = copy.deepcopy(self._mmTraces[activityType].data)
        for i in range(len(cellTrace)):
            cellTrace[i] = self.getCellIndices(cellTrace[i])

        return self.mmGetCellTracePlot(cellTrace, self.numberOfCells(), activityType, title, showReset, resetShading)
