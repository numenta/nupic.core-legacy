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

  //cout << _tm.mmPrettyPrintConnections() << endl;
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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.8, 0.7, 11, 11, 0.4, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.8, 0.7, 11, 11, 0.4, 0.0, 42);

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
  
  init();// { "initialPermanence": 0.2,
  //          "connectedPermanence" : 0.7,
  //          "permanenceIncrement" : 0.2 });
  _tm.initialize({ 100 }, 1, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "initialPermanence": 0.2,
  //          "connectedPermanence" : 0.7,
  //          "permanenceIncrement" : 0.2,
  //          "cellsPerColumn" : 4 });
  _tm.initialize({ 100 }, 4, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "initialPermanence": 0.2,
  //          "connectedPermanence" : 0.7,
  //          "permanenceIncrement" : 0.2 });
  _tm.initialize({ 100 }, 1, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4,
  //          "activationThreshold" : 8 });
  _tm.initialize({ 100 }, 4, 8, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.8, 0.7, 11, 11, 0.4, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4 });
  _tm.initialize({ 100 }, 4, 11, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
  
  init();// { "cellsPerColumn": 4,
  //          "activationThreshold" : 8 });
  _tm.initialize({ 100 }, 4, 8, 0.2, 0.7, 11, 11, 0.2, 0.0, 42);

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
