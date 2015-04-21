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
* Implementation for unit tests for TemporalMemory & MonitorMixin classes
*/

#include "MonitorMixinTest.hpp"


void MonitorMixinTest::setUp()
{
  _patternMachine = utils::ConsecutivePatternMachine();
  _patternMachine.initialize(100, vector<UInt>{ 5 });
  _sequenceMachine = utils::SequenceMachine(_patternMachine);

  _tm.initialize(vector<UInt>{ 100 }, 4);// , 0.6, 0.5, 1, 6, 0.1, 0.05, 1);
}

void MonitorMixinTest::RunTests()
{
  testFeedSequence();
  testClearHistory();
  testSequencesMetrics();

}

void MonitorMixinTest::testFeedSequence()
{
/*  sequence = self._generateSequence()
    sequenceLength = len(sequence) - 3  # without resets

    # Replace last pattern(before the None) with an unpredicted one
    sequence[-2] = self.patternMachine.get(4)

    self._feedSequence(sequence, sequenceLabel = "Test")

    activeColumnsTrace = self.tm.mmGetTraceActiveColumns()
    predictiveCellsTrace = self.tm.mmGetTracePredictiveCells()
    sequenceLabelsTrace = self.tm.mmGetTraceSequenceLabels()
    resetsTrace = self.tm.mmGetTraceResets()
    predictedActiveCellsTrace = self.tm.mmGetTracePredictedActiveCells()
    predictedInactiveCellsTrace = self.tm.mmGetTracePredictedInactiveCells()
    predictedActiveColumnsTrace = self.tm.mmGetTracePredictedActiveColumns()
    predictedInactiveColumnsTrace = self.tm.mmGetTracePredictedInactiveColumns()
    unpredictedActiveColumnsTrace = self.tm.mmGetTraceUnpredictedActiveColumns()

    self.assertEqual(len(activeColumnsTrace.data), sequenceLength)
    self.assertEqual(len(predictiveCellsTrace.data), sequenceLength)
    self.assertEqual(len(sequenceLabelsTrace.data), sequenceLength)
    self.assertEqual(len(resetsTrace.data), sequenceLength)
    self.assertEqual(len(predictedActiveCellsTrace.data), sequenceLength)
    self.assertEqual(len(predictedInactiveCellsTrace.data), sequenceLength)
    self.assertEqual(len(predictedActiveColumnsTrace.data), sequenceLength)
    self.assertEqual(len(predictedInactiveColumnsTrace.data), sequenceLength)
    self.assertEqual(len(unpredictedActiveColumnsTrace.data), sequenceLength)

    self.assertEqual(activeColumnsTrace.data[-1], self.patternMachine.get(4))
    self.assertEqual(sequenceLabelsTrace.data[-1], "Test")
    self.assertEqual(resetsTrace.data[0], True)
    self.assertEqual(resetsTrace.data[1], False)
    self.assertEqual(resetsTrace.data[10], True)
    self.assertEqual(resetsTrace.data[-1], False)
    self.assertEqual(len(predictedActiveCellsTrace.data[-2]), 5)
    self.assertEqual(len(predictedActiveCellsTrace.data[-1]), 0)
    self.assertEqual(len(predictedInactiveCellsTrace.data[-2]), 0)
    self.assertEqual(len(predictedInactiveCellsTrace.data[-1]), 5)
    self.assertEqual(len(predictedActiveColumnsTrace.data[-2]), 5)
    self.assertEqual(len(predictedActiveColumnsTrace.data[-1]), 0)
    self.assertEqual(len(predictedInactiveColumnsTrace.data[-2]), 0)
    self.assertEqual(len(predictedInactiveColumnsTrace.data[-1]), 5)
    self.assertEqual(len(unpredictedActiveColumnsTrace.data[-2]), 0)
    self.assertEqual(len(unpredictedActiveColumnsTrace.data[-1]), 5)
*/
}

void MonitorMixinTest::testClearHistory()
{
/*  sequence = self._generateSequence()
    self._feedSequence(sequence, sequenceLabel = "Test")
    self.tm.mmClearHistory()

    activeColumnsTrace = self.tm.mmGetTraceActiveColumns()
    predictiveCellsTrace = self.tm.mmGetTracePredictiveCells()
    sequenceLabelsTrace = self.tm.mmGetTraceSequenceLabels()
    resetsTrace = self.tm.mmGetTraceResets()
    predictedActiveCellsTrace = self.tm.mmGetTracePredictedActiveCells()
    predictedInactiveCellsTrace = self.tm.mmGetTracePredictedInactiveCells()
    predictedActiveColumnsTrace = self.tm.mmGetTracePredictedActiveColumns()
    predictedInactiveColumnsTrace = self.tm.mmGetTracePredictedInactiveColumns()
    unpredictedActiveColumnsTrace = self.tm.mmGetTraceUnpredictedActiveColumns()

    self.assertEqual(len(activeColumnsTrace.data), 0)
    self.assertEqual(len(predictiveCellsTrace.data), 0)
    self.assertEqual(len(sequenceLabelsTrace.data), 0)
    self.assertEqual(len(resetsTrace.data), 0)
    self.assertEqual(len(predictedActiveCellsTrace.data), 0)
    self.assertEqual(len(predictedInactiveCellsTrace.data), 0)
    self.assertEqual(len(predictedActiveColumnsTrace.data), 0)
    self.assertEqual(len(predictedInactiveColumnsTrace.data), 0)
    self.assertEqual(len(unpredictedActiveColumnsTrace.data), 0)
*/
}

void MonitorMixinTest::testSequencesMetrics()
{
/*  sequence = self._generateSequence()
    self._feedSequence(sequence, "Test1")

    sequence.reverse()
    sequence.append(sequence.pop(0))  # Move None(reset) to the end
    self._feedSequence(sequence, "Test2")

    sequencesPredictedActiveCellsPerColumnMetric = \
    self.tm.mmGetMetricSequencesPredictedActiveCellsPerColumn()
    sequencesPredictedActiveCellsSharedMetric = \
    self.tm.mmGetMetricSequencesPredictedActiveCellsShared()

    self.assertEqual(sequencesPredictedActiveCellsPerColumnMetric.mean, 1)
    self.assertEqual(sequencesPredictedActiveCellsSharedMetric.mean, 1)

    self._feedSequence(sequence, "Test3")

    sequencesPredictedActiveCellsPerColumnMetric = \
    self.tm.mmGetMetricSequencesPredictedActiveCellsPerColumn()
    sequencesPredictedActiveCellsSharedMetric = \
    self.tm.mmGetMetricSequencesPredictedActiveCellsShared()

    self.assertEqual(sequencesPredictedActiveCellsPerColumnMetric.mean, 1)
    self.assertTrue(sequencesPredictedActiveCellsSharedMetric.mean > 1)
*/
}


// ==============================
// Helper functions
// ==============================

void MonitorMixinTest::_generateSequence()
{
/*  numbers = range(0, 10)
    sequence = self.sequenceMachine.generateFromNumbers(numbers)
    sequence.append(None)
    sequence *= 3

    return sequence
*/
}

void MonitorMixinTest::_feedSequence(utils::Sequence& sequence, string sequenceLabel)
{
/*  for (vector<UInt> pattern : sequence)
  {
    if (pattern.size() == 0)
      _tm.reset();
    else
      _tm.compute(pattern, sequenceLabel);
  }
*/
}
