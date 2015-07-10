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
* Implementation of unit tests for TemporalMemory
*/

/*==============================================================================
                  Basic First Order Sequences
  ==============================================================================

  These tests ensure the most basic (first order) sequence learning mechanism is
  working.

  Parameters: Use a "fast learning mode": initPerm should be greater than
  connectedPerm and permanenceDec should be zero. With these settings sequences
  should be learned in one pass:

    minThreshold = newSynapseCount
    initialPermanence = 0.8
    connectedPermanence = 0.7
    permanenceDecrement = 0
    permanenceIncrement = 0.4

  Other Parameters:
    columnDimensions = [100]
    cellsPerColumn = 1
    newSynapseCount = 11
    activationThreshold = 11

  Note: this is not a high order sequence, so one cell per column is fine.

  Input Sequence: We train with M input sequences, each consisting of N random
  patterns. Each pattern consists of a random number of bits on. The number of
  1's in each pattern should be between 21 and 25 columns.

  Each input pattern can optionally have an amount of spatial noise represented
  by X, where X is the probability of switching an on bit with a random bit.

  Training: The TP is trained with P passes of the M sequences. There
  should be a reset between sequences. The total number of iterations during
  training is P*N*M.

  Testing: Run inference through the same set of sequences, with a reset before
  each sequence. For each sequence the system should accurately predict the
  pattern at the next time step up to and including the N-1'st pattern. The number
  of predicted inactive cells at each time step should be reasonably low.

  We can also calculate the number of synapses that should be
  learned. We raise an error if too many or too few were learned.

  B1) Basic sequence learner.  M=1, N=100, P=1.

  B2) Same as above, except P=2. Test that permanences go up and that no
  additional synapses are learned. [TODO]

  B3) N=300, M=1, P=1. (See how high we can go with N)

  B4) N=100, M=3, P=1. (See how high we can go with N*M)

  B5) Like B1 but with cellsPerColumn = 4. First order sequences should still
  work just fine.

  B6) Like B4 but with cellsPerColumn = 4. First order sequences should still
  work just fine.

  B7) Like B1 but with slower learning. Set the following parameters differently:

      initialPermanence = 0.2
      connectedPermanence = 0.7
      permanenceIncrement = 0.2

  Now we train the TP with the B1 sequence 4 times (P=4). This will increment
  the permanences to be above 0.8 and at that point the inference will be correct.
  This test will ensure the basic match function and segment activation rules are
  working correctly.

  B8) Like B7 but with 4 cells per column. Should still work.

  B9) Like B7 but present the sequence less than 4 times: the inference should be
  incorrect.

  B10) Like B2, except that cells per column = 4. Should still add zero additional
  synapses. [TODO]

  B11) Like B5, but with activationThreshold = 8 and with each pattern
  corrupted by a small amount of spatial noise (X = 0.05).


  ===============================================================================
                  High Order Sequences
  ===============================================================================

  These tests ensure that high order sequences can be learned in a multiple cells
  per column instantiation.

  Parameters: Same as Basic First Order Tests above, but with varying cells per
  column.

  Input Sequence: We train with M input sequences, each consisting of N random
  patterns. Each pattern consists of a random number of bits on. The number of
  1's in each pattern should be between 21 and 25 columns. The sequences are
  constructed to contain shared subsequences, such as:

  A B C D E F G H I J
  K L M D E F N O P Q

  The position and length of shared subsequences are parameters in the tests.

  Each input pattern can optionally have an amount of spatial noise represented
  by X, where X is the probability of switching an on bit with a random bit.

  Training: Identical to basic first order tests above.

  Testing: Identical to basic first order tests above unless noted.

  We can also calculate the number of segments and synapses that should be
  learned. We raise an error if too many or too few were learned.

  H1) Learn two sequences with a shared subsequence in the middle. Parameters
  should be the same as B1. Since cellsPerColumn == 1, it should make more
  predictions than necessary.

  H2) Same as H1, but with cellsPerColumn == 4, and train multiple times.
  It should make just the right number of predictions.

  H3) Like H2, except the shared subsequence is in the beginning (e.g.
  "ABCDEF" and "ABCGHIJ"). At the point where the shared subsequence ends, all
  possible next patterns should be predicted. As soon as you see the first unique
  pattern, the predictions should collapse to be a perfect prediction.

  H4) Shared patterns. Similar to H2 except that patterns are shared between
  sequences.  All sequences are different shufflings of the same set of N
  patterns (there is no shared subsequence).

  H5) Combination of H4) and H2). Shared patterns in different sequences, with a
  shared subsequence.

  H6) Stress test: every other pattern is shared. [TODO]

  H7) Start predicting in the middle of a sequence. [TODO]

  H8) Hub capacity. How many patterns can use that hub? [TODO]

  H9) Sensitivity to small amounts of spatial noise during inference (X = 0.05).
  Parameters the same as B11, and sequences like H2.

  H10) Higher order patterns with alternating elements.

  Create the following 4 sequences:

       A B A B A C
       A B A B D E
       A B F G H I
       A J K L M N

  After training we should verify that the expected transitions are in the
  model. Prediction accuracy should be perfect. In addition, during inference,
  after the first element is presented, the columns should not burst any more.
  Need to verify, for the first sequence, that the high order representation
  when presented with the second A and B is different from the representation
  in the first presentation. [TODO]
  """
  */
#include "TemporalMemoryExtensiveTest.hpp"

void TemporalMemoryExtensiveTest::init()
{
//    DEFAULT_TM_PARAMS = {
//      "columnDimensions": [100],
//      "cellsPerColumn" : 1,
//      "initialPermanence" : 0.8,
//      "connectedPermanence" : 0.7,
//      "minThreshold" : 11,
//      "maxNewSynapseCount" : 11,
//      "permanenceIncrement" : 0.4,
//      "permanenceDecrement" : 0,
//      "activationThreshold" : 11
//  }

  _verbosity = 1;

  TemporalMemoryAbstractTest::init();
}

//==============================
// Overrides
// ==============================

void TemporalMemoryExtensiveTest::setUp()
{
  TemporalMemoryAbstractTest::setUp();

  _patternMachine = PatternMachine();
  _patternMachine.initialize(100, range(21, 26), 300);
  _sequenceMachine = SequenceMachine(_patternMachine);

  cout << endl;
  cout << "======================================================" << endl;
  cout << "Test: ";// << id() << endl;
//  cout << shortDescription() << endl;
  cout << "======================================================" << endl;
}


void TemporalMemoryExtensiveTest::_feedTM(Sequence& sequence, bool learn, int num)
{
  TemporalMemoryAbstractTest::_feedTM(sequence, learn, num);

  if (_verbosity >= 2)
  {
    _tm.mmPrettyPrintTraces(_tm.mmGetDefaultTraces(_verbosity - 1), _tm.mmGetTraceResets());
    cout << endl;
  }
  if (learn && _verbosity >= 3)
    cout << _tm.mmPrettyPrintConnections();
}


// ==============================
// Helper functions
// ==============================

void TemporalMemoryExtensiveTest::_testTM(Sequence& sequence)
{
  _feedTM(sequence, false);

  cout << _tm.mmPrettyPrintMetrics(_tm.mmGetDefaultMetrics());
}


void TemporalMemoryExtensiveTest::assertAllActiveWerePredicted()
{
  MetricsVector unpredictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTraceUnpredictedActiveColumns());
  MetricsVector predictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedActiveColumns());

  NTA_CHECK(unpredictedActiveColumnsMetric._sum == 0);

  NTA_CHECK(predictedActiveColumnsMetric._min == 21);
  NTA_CHECK(predictedActiveColumnsMetric._max == 25);
}


void TemporalMemoryExtensiveTest::assertAllInactiveWereUnpredicted()
{
  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());

  NTA_CHECK(predictedInactiveColumnsMetric._sum == 0);
}


void TemporalMemoryExtensiveTest::assertAllActiveWereUnpredicted()
{
  MetricsVector unpredictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTraceUnpredictedActiveColumns());
  MetricsVector predictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedActiveColumns());

  NTA_CHECK(predictedActiveColumnsMetric._sum == 0);

  NTA_CHECK(unpredictedActiveColumnsMetric._min == 21);
  NTA_CHECK(unpredictedActiveColumnsMetric._max == 25);
}


void TemporalMemoryExtensiveTest::testB1()
{
  // Basic sequence learner.  M=1, N=100, P=1.
  init();

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  // Clear this sequence and re-populate with data from Python (via PyCharm debugger)
  sequence.data.clear();
  sequence.data.push_back({ 96, 1, 66, 35, 33, 16, 71, 10, 43, 44, 79, 48, 49, 19, 20, 89, 52, 58, 63, 74, 31 });
  sequence.data.push_back({ 0, 1, 4, 10, 15, 16, 23, 28, 31, 40, 46, 48, 50, 58, 62, 67, 71, 77, 79, 90, 92, 93, 97, 99 });
  sequence.data.push_back({ 5, 7, 10, 14, 17, 21, 24, 29, 31, 34, 39, 46, 53, 59, 61, 67, 69, 71, 77, 81, 85, 87, 88, 93, 97 });
  sequence.data.push_back({ 5, 6, 10, 11, 14, 17, 26, 30, 33, 38, 39, 42, 45, 52, 53, 58, 59, 60, 64, 67, 74, 97, 98, 99 });
  sequence.data.push_back({ 5, 13, 19, 23, 25, 26, 29, 35, 36, 37, 42, 48, 57, 59, 60, 67, 73, 77, 92, 95, 96, 98 });
  sequence.data.push_back({ 3, 17, 18, 21, 24, 27, 34, 35, 37, 40, 44, 47, 50, 58, 59, 60, 62, 64, 66, 69, 77, 79, 82, 95, 98 });
  sequence.data.push_back({ 1, 5, 15, 18, 24, 32, 37, 39, 40, 42, 45, 58, 64, 68, 69, 72, 77, 83, 87, 90, 91, 98 });
  sequence.data.push_back({ 1, 12, 15, 22, 23, 24, 25, 36, 37, 38, 50, 55, 56, 63, 65, 66, 70, 72, 74, 81, 82, 87, 92, 95, 99 });
  sequence.data.push_back({ 4, 5, 7, 11, 14, 16, 40, 45, 46, 47, 53, 58, 60, 67, 69, 71, 73, 75, 77, 79, 85, 89, 95, 96 });
  sequence.data.push_back({ 0, 4, 10, 12, 18, 22, 26, 30, 31, 33, 39, 44, 45, 53, 55, 70, 73, 76, 77, 80, 83, 88, 90 });
  sequence.data.push_back({ 4, 7, 11, 12, 14, 19, 20, 23, 26, 27, 31, 35, 40, 50, 54, 56, 58, 65, 75, 81, 84, 85, 87, 97, 98 });
  sequence.data.push_back({ 96, 65, 66, 68, 33, 40, 76, 75, 12, 45, 78, 16, 24, 19, 20, 53, 56, 89, 71, 62, 63 });
  sequence.data.push_back({ 1, 3, 7, 11, 16, 18, 19, 22, 29, 31, 33, 34, 37, 43, 50, 53, 60, 61, 62, 72, 74, 77, 80, 91 });
  sequence.data.push_back({ 1, 8, 10, 20, 22, 23, 27, 29, 30, 31, 41, 51, 55, 61, 71, 74, 76, 80, 81, 86, 93, 94 });
  sequence.data.push_back({ 4, 11, 12, 14, 20, 21, 26, 31, 34, 38, 46, 47, 51, 52, 57, 64, 66, 75, 79, 90, 92, 93, 94, 99 });
  sequence.data.push_back({ 0, 5, 7, 17, 19, 24, 25, 38, 40, 43, 44, 45, 46, 48, 54, 55, 66, 74, 77, 86, 90, 95 });
  sequence.data.push_back({ 3, 6, 19, 31, 40, 41, 44, 46, 47, 53, 55, 67, 68, 70, 73, 75, 76, 83, 88, 90, 97, 99 });
  sequence.data.push_back({ 4, 9, 16, 19, 21, 23, 37, 43, 45, 46, 47, 55, 62, 72, 74, 79, 81, 89, 93, 94, 98, 99 });
  sequence.data.push_back({ 1, 34, 66, 38, 6, 71, 8, 2, 12, 45, 14, 47, 49, 82, 54, 56, 36, 26, 59, 60, 61 });
  sequence.data.push_back({ 5, 6, 7, 11, 17, 22, 23, 25, 27, 28, 30, 35, 38, 42, 52, 66, 67, 68, 75, 78, 82, 87, 88, 93, 99 });
  sequence.data.push_back({ 64, 51, 35, 68, 70, 6, 71, 96, 75, 76, 15, 80, 17, 19, 53, 23, 4, 58, 93, 30, 5 });
  sequence.data.push_back({ 0, 11, 12, 13, 14, 17, 18, 21, 23, 24, 26, 33, 37, 62, 71, 77, 80, 81, 84, 89, 90, 93, 96 });
  sequence.data.push_back({ 0, 34, 99, 69, 6, 39, 31, 11, 44, 77, 78, 81, 82, 20, 22, 55, 23, 52, 91, 62, 95 });
  sequence.data.push_back({ 3, 9, 18, 19, 21, 28, 29, 34, 39, 42, 45, 54, 61, 62, 65, 66, 72, 74, 80, 91, 95, 97, 99 });
  sequence.data.push_back({ 4, 8, 9, 11, 13, 15, 16, 35, 36, 37, 39, 49, 50, 56, 66, 67, 68, 72, 78, 84, 91, 98, 99 });
  sequence.data.push_back({ 1, 10, 13, 17, 18, 19, 28, 29, 30, 31, 32, 35, 36, 37, 38, 44, 66, 71, 75, 81, 84, 90 });
  sequence.data.push_back({ 0, 64, 2, 99, 4, 70, 38, 39, 18, 45, 78, 50, 52, 85, 87, 57, 90, 63, 84, 93, 95 });
  sequence.data.push_back({ 6, 9, 11, 13, 15, 17, 26, 27, 29, 33, 35, 37, 39, 48, 54, 55, 57, 62, 63, 75, 84, 89, 95, 98 });
  sequence.data.push_back({ 3, 4, 7, 8, 9, 10, 18, 21, 30, 36, 38, 47, 49, 51, 55, 56, 60, 67, 68, 71, 74, 80, 82, 91, 95 });
  sequence.data.push_back({ 2, 4, 9, 17, 19, 26, 29, 31, 34, 47, 49, 59, 61, 62, 63, 64, 65, 74, 80, 88, 91, 98 });
  sequence.data.push_back({ 4, 14, 18, 20, 21, 25, 28, 35, 38, 44, 50, 51, 59, 60, 62, 63, 70, 73, 75, 79, 85, 93, 97 });
  sequence.data.push_back({ 2, 3, 10, 12, 17, 19, 22, 26, 28, 35, 37, 41, 45, 47, 48, 50, 51, 64, 66, 82, 85, 86, 88, 94, 97 });
  sequence.data.push_back({ 5, 8, 12, 14, 18, 19, 22, 23, 27, 39, 40, 44, 48, 49, 50, 54, 60, 61, 68, 71, 79, 87, 91, 97 });
  sequence.data.push_back({ 6, 10, 25, 29, 33, 34, 40, 41, 43, 46, 57, 58, 65, 71, 75, 77, 78, 79, 84, 91, 92, 93, 99 });
  sequence.data.push_back({ 4, 5, 6, 8, 12, 13, 20, 25, 26, 29, 30, 35, 36, 47, 57, 60, 67, 74, 76, 85, 91, 92, 94, 97 });
  sequence.data.push_back({ 0, 4, 5, 9, 12, 17, 25, 34, 38, 39, 41, 42, 44, 56, 65, 68, 69, 74, 76, 77, 83, 88, 93, 96, 99 });
  sequence.data.push_back({ 41, 48, 67, 17, 69, 70, 7, 9, 10, 11, 44, 77, 16, 76, 75, 19, 53, 86, 73, 35, 85 });
  sequence.data.push_back({ 0, 97, 66, 35, 33, 72, 28, 10, 11, 78, 49, 50, 19, 57, 54, 73, 83, 89, 60, 29, 63 });
  sequence.data.push_back({ 5, 6, 8, 12, 17, 25, 29, 31, 33, 34, 35, 46, 53, 56, 57, 63, 67, 71, 78, 79, 87, 88, 95 });
  sequence.data.push_back({ 3, 9, 17, 19, 23, 27, 29, 30, 32, 35, 36, 40, 52, 56, 58, 60, 63, 67, 79, 83, 85, 88, 90, 95 });
  sequence.data.push_back({ 32, 1, 66, 36, 20, 6, 39, 64, 44, 82, 14, 45, 81, 18, 83, 52, 55, 89, 59, 30, 31 });
  sequence.data.push_back({ 1, 17, 21, 24, 28, 31, 33, 34, 37, 39, 45, 48, 53, 54, 58, 59, 66, 74, 76, 80, 86, 88, 93, 97 });
  sequence.data.push_back({ 15, 18, 22, 27, 29, 32, 33, 35, 36, 39, 40, 42, 44, 53, 58, 69, 79, 80, 83, 86, 91, 95, 97 });
  sequence.data.push_back({ 0, 65, 98, 75, 66, 39, 40, 12, 7, 76, 45, 14, 79, 81, 67, 25, 26, 27, 92, 44, 63 });
  sequence.data.push_back({ 0, 1, 2, 6, 7, 9, 20, 28, 29, 30, 38, 46, 47, 51, 55, 58, 60, 66, 69, 87, 94, 97 });
  sequence.data.push_back({ 5, 7, 12, 14, 16, 19, 22, 23, 27, 31, 32, 41, 43, 53, 57, 65, 68, 73, 75, 78, 82, 91, 92, 96, 99 });
  sequence.data.push_back({ 5, 9, 12, 15, 18, 19, 21, 26, 36, 37, 40, 49, 50, 51, 63, 66, 69, 76, 78, 81, 91, 93 });
  sequence.data.push_back({ 1, 2, 3, 4, 8, 9, 13, 22, 25, 32, 36, 37, 46, 49, 52, 53, 56, 61, 64, 84, 90, 91, 93, 96 });
  sequence.data.push_back({ 1, 3, 6, 7, 11, 12, 17, 18, 21, 25, 30, 33, 45, 50, 75, 76, 77, 78, 79, 81, 84, 91, 95 });
  sequence.data.push_back({ 1, 10, 12, 18, 26, 38, 40, 49, 53, 55, 61, 65, 66, 68, 71, 77, 80, 81, 89, 90, 92, 96 });
  sequence.data.push_back({ 3, 12, 13, 16, 18, 20, 24, 26, 30, 31, 35, 38, 42, 50, 52, 54, 55, 56, 59, 60, 61, 67, 77, 79, 98 });
  sequence.data.push_back({ 3, 4, 5, 9, 11, 13, 16, 19, 21, 22, 24, 48, 49, 60, 61, 65, 66, 69, 70, 78, 85, 87, 96 });
  sequence.data.push_back({ 67, 35, 4, 6, 92, 41, 43, 76, 77, 78, 47, 48, 19, 21, 86, 55, 58, 27, 60, 94, 95 });
  sequence.data.push_back({ 1, 3, 5, 10, 16, 23, 30, 35, 39, 46, 47, 50, 52, 54, 60, 68, 72, 79, 83, 84, 85, 88, 92, 99 });
  sequence.data.push_back({ 2, 6, 8, 13, 15, 16, 27, 33, 37, 41, 48, 52, 61, 62, 64, 68, 73, 74, 75, 79, 89, 91, 96, 98 });
  sequence.data.push_back({ 2, 3, 4, 14, 15, 19, 27, 28, 31, 35, 45, 48, 56, 58, 63, 69, 71, 73, 78, 80, 90, 98 });
  sequence.data.push_back({ 1, 3, 14, 17, 19, 22, 25, 26, 27, 28, 35, 36, 48, 49, 55, 58, 67, 69, 72, 80, 85, 93, 98 });
  sequence.data.push_back({ 96, 97, 2, 99, 68, 37, 7, 73, 74, 75, 12, 66, 81, 51, 84, 23, 89, 90, 60, 29, 63 });
  sequence.data.push_back({ 10, 14, 24, 25, 30, 37, 40, 44, 56, 58, 60, 63, 70, 72, 74, 80, 82, 84, 86, 93, 97, 99 });
  sequence.data.push_back({ 84, 97, 67, 4, 16, 71, 72, 95, 7, 12, 13, 46, 80, 49, 20, 53, 23, 88, 89, 27, 31 });
  sequence.data.push_back({ 0, 97, 83, 96, 6, 7, 73, 10, 43, 76, 12, 16, 18, 19, 84, 54, 23, 88, 92, 29, 31 });
  sequence.data.push_back({ 84, 97, 61, 67, 68, 69, 7, 11, 12, 77, 79, 75, 52, 86, 23, 25, 33, 5, 29, 62, 95 });
  sequence.data.push_back({ 3, 34, 27, 37, 73, 14, 82, 46, 13, 92, 49, 50, 19, 52, 53, 41, 25, 47, 21, 42, 63 });
  sequence.data.push_back({ 0, 1, 10, 11, 22, 30, 36, 44, 47, 53, 56, 57, 58, 59, 65, 66, 75, 76, 84, 85, 90, 95, 98 });
  sequence.data.push_back({ 2, 10, 13, 15, 17, 19, 20, 24, 26, 38, 39, 40, 43, 46, 51, 58, 59, 61, 74, 78, 81, 82 });
  sequence.data.push_back({ 3, 4, 7, 15, 21, 26, 45, 47, 48, 50, 52, 60, 62, 64, 72, 77, 86, 90, 91, 94, 95, 98 });
  sequence.data.push_back({ 0, 2, 5, 6, 8, 11, 22, 23, 25, 30, 34, 37, 43, 49, 52, 53, 57, 61, 63, 68, 79, 90, 98 });
  sequence.data.push_back({ 7, 10, 11, 12, 18, 21, 22, 24, 35, 36, 41, 45, 46, 47, 49, 52, 55, 58, 60, 64, 74, 81, 85, 90, 94 });
  sequence.data.push_back({ 2, 3, 6, 8, 13, 24, 25, 29, 33, 44, 45, 53, 55, 60, 61, 64, 65, 76, 79, 80, 84, 90, 91, 94, 97 });
  sequence.data.push_back({ 0, 5, 10, 15, 25, 30, 31, 32, 33, 37, 43, 47, 48, 49, 58, 59, 60, 66, 67, 79, 83, 93, 97 });
  sequence.data.push_back({ 8, 11, 15, 20, 23, 31, 37, 46, 55, 57, 58, 60, 68, 70, 73, 74, 80, 82, 83, 86, 94, 99 });
  sequence.data.push_back({ 0, 4, 6, 8, 10, 11, 18, 22, 30, 35, 37, 43, 44, 47, 53, 58, 60, 61, 66, 69, 74, 75, 80, 90, 94 });
  sequence.data.push_back({ 0, 2, 7, 11, 14, 19, 20, 21, 28, 33, 35, 37, 43, 53, 64, 68, 69, 72, 73, 79, 80, 83, 86, 98 });
  sequence.data.push_back({ 96, 59, 36, 69, 71, 72, 42, 75, 45, 78, 11, 50, 57, 55, 24, 23, 90, 95, 5, 74, 37 });
  sequence.data.push_back({ 5, 19, 22, 26, 37, 38, 40, 41, 45, 48, 49, 51, 55, 58, 60, 64, 70, 74, 75, 78, 89, 92, 96 });
  sequence.data.push_back({ 7, 16, 18, 26, 30, 35, 41, 42, 43, 45, 48, 52, 53, 54, 71, 74, 78, 84, 85, 86, 90, 94, 99 });
  sequence.data.push_back({ 2, 4, 11, 19, 29, 30, 39, 45, 46, 48, 51, 52, 56, 59, 60, 61, 70, 74, 80, 89, 94, 96, 98, 99 });
  sequence.data.push_back({ 5, 7, 15, 22, 23, 25, 27, 40, 42, 47, 49, 56, 58, 59, 61, 62, 69, 77, 82, 89, 92, 96 });
  sequence.data.push_back({ 4, 5, 12, 15, 17, 20, 21, 23, 29, 32, 37, 43, 54, 60, 70, 71, 74, 77, 83, 88, 93, 96 });
  sequence.data.push_back({ 0, 11, 13, 18, 19, 22, 29, 30, 31, 32, 36, 42, 44, 50, 52, 63, 72, 79, 91, 92, 96, 99 });
  sequence.data.push_back({ 7, 10, 18, 19, 22, 23, 28, 29, 34, 37, 39, 46, 50, 51, 55, 56, 67, 68, 70, 71, 72, 80, 89, 91, 95 });
  sequence.data.push_back({ 0, 1, 3, 9, 10, 12, 21, 24, 32, 42, 48, 52, 53, 54, 62, 64, 67, 69, 73, 75, 78, 80, 81, 90, 99 });
  sequence.data.push_back({ 1, 4, 8, 9, 12, 13, 15, 23, 24, 29, 35, 45, 49, 51, 55, 59, 64, 65, 67, 72, 78, 82, 86, 87, 91 });
  sequence.data.push_back({ 1, 7, 12, 21, 23, 25, 27, 28, 36, 38, 43, 50, 51, 61, 66, 70, 75, 83, 84, 94, 97, 99 });
  sequence.data.push_back({ 2, 4, 6, 8, 11, 13, 15, 16, 23, 29, 38, 42, 46, 48, 51, 55, 56, 59, 65, 67, 76, 77, 78, 82, 89 });
  sequence.data.push_back({ 0, 20, 22, 23, 24, 25, 26, 27, 28, 31, 32, 37, 38, 41, 46, 47, 52, 56, 73, 81, 82, 83, 94, 97, 98 });
  sequence.data.push_back({ 6, 10, 21, 24, 26, 30, 36, 45, 47, 59, 62, 67, 68, 70, 71, 72, 75, 80, 81, 83, 91, 92, 95 });
  sequence.data.push_back({ 2, 7, 10, 12, 19, 25, 38, 39, 44, 46, 49, 52, 56, 72, 77, 78, 79, 83, 84, 88, 94, 97, 98, 99 });
  sequence.data.push_back({ 1, 7, 12, 13, 22, 25, 26, 31, 38, 40, 44, 58, 61, 62, 67, 70, 71, 73, 84, 85, 92, 97, 98 });
  sequence.data.push_back({ 1, 5, 7, 9, 10, 14, 16, 18, 22, 23, 30, 39, 42, 48, 49, 56, 58, 68, 77, 87, 95, 96 });
  sequence.data.push_back({ 0, 2, 3, 7, 10, 14, 16, 19, 21, 27, 28, 29, 35, 38, 40, 43, 49, 52, 55, 60, 62, 65, 82, 93, 96 });
  sequence.data.push_back({ 4, 15, 19, 23, 33, 34, 35, 37, 44, 46, 57, 59, 62, 69, 70, 71, 73, 77, 81, 82, 86, 91 });
  sequence.data.push_back({ 0, 2, 67, 69, 49, 73, 12, 45, 28, 16, 76, 18, 83, 84, 41, 23, 56, 63, 5, 62, 31 });
  sequence.data.push_back({ 0, 1, 11, 13, 17, 22, 26, 33, 36, 38, 41, 44, 61, 62, 63, 77, 80, 81, 85, 86, 90, 94 });
  sequence.data.push_back({ 4, 10, 29, 36, 43, 44, 50, 53, 57, 58, 63, 64, 66, 72, 75, 78, 79, 80, 86, 91, 97, 99 });
  sequence.data.push_back({ 0, 1, 10, 12, 15, 17, 24, 26, 28, 36, 39, 51, 52, 54, 55, 58, 63, 73, 77, 79, 83, 85, 87, 89, 94 });
  sequence.data.push_back({ 15, 17, 19, 23, 26, 27, 33, 34, 45, 46, 47, 49, 51, 53, 62, 63, 68, 69, 72, 74, 80, 81, 84, 94 });
  sequence.data.push_back({ 96, 2, 3, 4, 40, 73, 12, 61, 78, 47, 16, 50, 51, 46, 86, 26, 44, 28, 90, 30, 95 });
  sequence.data.push_back({ 1, 3, 5, 8, 14, 15, 22, 23, 26, 33, 35, 43, 48, 59, 61, 68, 74, 77, 79, 80, 90, 98 });
  sequence.data.push_back({ 1, 3, 13, 24, 25, 26, 30, 33, 37, 43, 49, 58, 63, 66, 67, 71, 76, 77, 78, 79, 90, 91, 97 });
  sequence.data.push_back({ });

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB3()
{
  // N=300, M=1, P=1. (See how high we can go with N)
  init();

  Sequence numbers = _sequenceMachine.generateNumbers(1, 300);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB4()
{
  // N=100, M=3, P=1. (See how high we can go with N*M)
  init();

  Sequence numbers = _sequenceMachine.generateNumbers(3, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
}


void TemporalMemoryExtensiveTest::testB5()
{
  // Like B1 but with cellsPerColumn = 4.
  // First order sequences should still work just fine.
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB6()
{
  // Like B4 but with cellsPerColumn = 4.
  // First order sequences should still work just fine."""
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers = _sequenceMachine.generateNumbers(3, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB7()
{
  // Like B1 but with slower learning.
  // 
  // Set the following parameters differently:
  // 
  //   initialPermanence = 0.2
  //   connectedPermanence = 0.7
  //   permanenceIncrement = 0.2
  // 
  // Now we train the TP with the B1 sequence 4 times (P=4). This will increment
  // the permanences to be above 0.8 and at that point the inference will be correct.
  // This test will ensure the basic match function and segment activation rules are
  // working correctly.
  
  //init({ "initialPermanence": 0.2,
  //       "connectedPermanence" : 0.7,
  //       "permanenceIncrement" : 0.2 });

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 4; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB8()
{
  // Like B7 but with 4 cells per column.
  // Should still work.
  
  //init({ "initialPermanence": 0.2,
  //       "connectedPermanence" : 0.7,
  //       "permanenceIncrement" : 0.2,
  //       "cellsPerColumn" : 4 });

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 4; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();
  assertAllInactiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB9()
{
  // Like B7 but present the sequence less than 4 times.
  // The inference should be incorrect.
  
  //init({ "initialPermanence": 0.2,
  //       "connectedPermanence" : 0.7,
  //       "permanenceIncrement" : 0.2 });

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 4; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWereUnpredicted();
}


void TemporalMemoryExtensiveTest::testB11()
{
  // Like B5, but with activationThreshold = 8 and with each pattern
  // corrupted by a small amount of spatial noise (X = 0.05).
  
  //init({ "cellsPerColumn": 4,
  //       "activationThreshold" : 8 });

  Sequence numbers = _sequenceMachine.generateNumbers(1, 100);
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  sequence = _sequenceMachine.addSpatialNoise(sequence, 0.05);

  _testTM(sequence);
  MetricsVector unpredictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTraceUnpredictedActiveColumns());
  NTA_CHECK(unpredictedActiveColumnsMetric._mean < 1);
}


void TemporalMemoryExtensiveTest::testH1()
{
  // Learn two sequences with a short shared pattern.
  // Parameters should be the same as B1.
  // Since cellsPerColumn == 1, it should make more predictions than necessary.
  init();

  Sequence numbers = _sequenceMachine.generateNumbers(2, 20, { 10, 15 });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();

  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());
  NTA_CHECK(predictedInactiveColumnsMetric._mean > 0);


  // At the end of both shared sequences, there should be
  // predicted but inactive columns
  NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[15].size() > 0);
  NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[35].size() > 0);
}


void TemporalMemoryExtensiveTest::testH2()
{
  // Same as H1, but with cellsPerColumn == 4, and train multiple times.
  // It should make just the right number of predictions.
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers = _sequenceMachine.generateNumbers(2, 20, { 10, 15 });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 10; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();

  // Without some kind of decay, expect predicted inactive columns at the
  // end of the first shared sequence
  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());
  NTA_CHECK(predictedInactiveColumnsMetric._sum < 26);

  // At the end of the second shared sequence, there should be no
  // predicted but inactive columns
  NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[36].size() == 0);
}


void TemporalMemoryExtensiveTest::testH3()
{
  // Like H2, except the shared subsequence is in the beginning.
  // (e.g. "ABCDEF" and "ABCGHIJ") At the point where the shared subsequence
  // ends, all possible next patterns should be predicted.As soon as you see
  // the first unique pattern, the predictions should collapse to be a perfect
  // prediction.
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers = _sequenceMachine.generateNumbers(2, 20, { 0, 5 });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();

  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());
  NTA_CHECK(predictedInactiveColumnsMetric._sum < 26 * 2);

  // At the end of each shared sequence, there should be
  // predicted but inactive columns
  NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[5].size() > 0);
  NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[25].size() > 0);
}


void TemporalMemoryExtensiveTest::testH4()
{
  // Shared patterns. Similar to H2 except that patterns are shared between
  // sequences. All sequences are different shufflings of the same set of N
  // patterns (there is no shared subsequence).
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers;
  numbers += _sequenceMachine.generateNumbers(1, 20);
  numbers += _sequenceMachine.generateNumbers(1, 20);

  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 20; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();

  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());
  NTA_CHECK(predictedInactiveColumnsMetric._mean < 3);
}


void TemporalMemoryExtensiveTest::testH5()
{
  // Combination of H4) and H2).
  // Shared patterns in different sequences, with a shared subsequence.
  
  //init({ "cellsPerColumn": 4 });

  Sequence numbers;
  Sequence shared = _sequenceMachine.generateNumbers(1, 5);// [:-1];
  for (int i = 0; i < 2; i++)
  {
    Sequence sublist = _sequenceMachine.generateNumbers(1, 20);
    //sublist = [x for x in sublist if x not in xrange(5)];
    
    numbers += sublist;// [0:10];
    numbers += shared;
    numbers += sublist;// [10:];
  }

  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 20; i++)
    _feedTM(sequence);

  _testTM(sequence);
  assertAllActiveWerePredicted();

  MetricsVector predictedInactiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTracePredictedInactiveColumns());
  NTA_CHECK(predictedInactiveColumnsMetric._mean < 3);
}


void TemporalMemoryExtensiveTest::testH9()
{
  // Sensitivity to small amounts of spatial noise during inference
  // (X = 0.05). Parameters the same as B11, and sequences like H2.
  
  //init({ "cellsPerColumn": 4,
  //       "activationThreshold" : 8 });

  Sequence numbers = _sequenceMachine.generateNumbers(2, 20, { 10, 15 });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 10; i++)
    _feedTM(sequence);

  sequence = _sequenceMachine.addSpatialNoise(sequence, 0.05);

  _testTM(sequence);
  MetricsVector unpredictedActiveColumnsMetric = _tm.mmGetMetricFromTrace(
    _tm.mmGetTraceUnpredictedActiveColumns());
  NTA_CHECK(unpredictedActiveColumnsMetric._mean < 3);
}
