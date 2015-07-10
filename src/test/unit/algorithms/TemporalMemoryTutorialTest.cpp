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

#include "TemporalMemoryTutorialTest.hpp"


void TemporalMemoryTutorialTest::testFirstOrder()
{
  // Basic first order sequences
  init();

  Sequence numbers;
  numbers.push_back(vector<UInt>{ 0, 1, 2, 3 });
  numbers.push_back({});
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  _feedTM(sequence);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 0);

  _feedTM(sequence, 2);

  _feedTM(sequence);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);

  _feedTM(sequence, 4);

  _feedTM(sequence);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);
}


void TemporalMemoryTutorialTest::testHighOrder()
{
  // High order sequences (in order)
  init();

  Sequence numbersA(vector<vector<UInt>>{ {0, 1, 2, 3}, {} });
  Sequence sequenceA = _sequenceMachine.generateFromNumbers(numbersA);
  Sequence numbersB(vector<vector<UInt>>{ {4, 1, 2, 5}, {} });
  Sequence sequenceB = _sequenceMachine.generateFromNumbers(numbersB);

  _feedTM(sequenceA, 5);

  _feedTM(sequenceA, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);

  _feedTM(sequenceB);

  _feedTM(sequenceB, 2);

  _feedTM(sequenceB, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[1].size() == 1);

  _feedTM(sequenceB, 3);

  _feedTM(sequenceB, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[2].size() == 1);

  _feedTM(sequenceB, 3);

  _feedTM(sequenceB, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);

  _feedTM(sequenceA, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);
  //NTA_CHECK(_tm.mmGetTracePredictedInactiveColumns()._data[3].size() == 1);

  _feedTM(sequenceA, 10);
  _feedTM(sequenceA, 1, false);
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);

  // TODO: Requires some form of synaptic decay to forget the ABC = > Y
  // transition that's initially formed
  //NTA_CHECK(len(self.tm.mmGetTracePredictedInactiveColumns()._data[3].size() == 0);
}


void TemporalMemoryTutorialTest::testHighOrderAlternating()
{
  // High order sequences (alternating)
  init();

  Sequence numbersA( { { 0, 1, 2, 3 },{} } );
  Sequence numbersB( { { 4, 1, 2, 5 },{} } );

  Sequence sequence = _sequenceMachine.generateFromNumbers(numbersA);
  sequence += _sequenceMachine.generateFromNumbers(numbersB);

  _feedTM(sequence);
  _feedTM(sequence, 10);
  _feedTM(sequence, 1, false);

  // TODO: Requires some form of synaptic decay to forget the
  // ABC = > Y and XBC = > D transitions that are initially formed
  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[3].size() == 1);
  // NTA_CHECK(len(self.tm.mmGetTracePredictedInactiveColumns()._data[3]), 0)

  //NTA_CHECK(_tm.mmGetTracePredictedActiveColumns()._data[7].size() == 1);
  // NTA_CHECK(len(self.tm.mmGetTracePredictedInactiveColumns()._data[7]), 0)
}


void TemporalMemoryTutorialTest::testEndlesslyRepeating()
{
  // Endlessly repeating sequence of 2 elements
  init();// { "columnDimensions": [2] });
  _tm.initialize({ 2 }, 4, 1, 0.3, 0.5, 1, 6, 0.1, 0.05, 42);

  Sequence numbers({ { 0, 1 } });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 7; i++)
    _feedTM(sequence);

  _feedTM(sequence, 50);

}


void TemporalMemoryTutorialTest::testEndlesslyRepeatingWithNoNewSynapses()
{
  // Endlessly repeating sequence of 2 elements with maxNewSynapseCount=1"""
  
  init();// { "columnDimensions": [2],
  //          "maxNewSynapseCount" : 1,
  //          "cellsPerColumn" : 10 });
  _tm.initialize({ 6 }, 10, 1, 0.3, 0.5, 1, 1, 0.1, 0.05, 42);

  Sequence numbers({ { 0, 1 } });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);

  for (int i = 0; i < 7; i++)
    _feedTM(sequence);

  _feedTM(sequence, 100);

}


void TemporalMemoryTutorialTest::testLongRepeatingWithNovelEnding()
{
  // Long repeating sequence with novel pattern at the end
  
  init();// { "columnDimensions": [3] });
  _tm.initialize({ 3 }, 4, 1, 0.3, 0.5, 1, 6, 0.1, 0.05, 42);

  Sequence numbers({ { 0, 1 } });
  Sequence sequence = _sequenceMachine.generateFromNumbers(numbers);
  sequence *= 10;
//  sequence += {_patternMachine.get(2), {}};

  for (int i = 0; i < 4; i++)
    _feedTM(sequence);

  _feedTM(sequence, 10);
}


void TemporalMemoryTutorialTest::testSingleEndlesslyRepeating()
{
  // A single endlessly repeating pattern
  
  init();// { "columnDimensions": [1] });
  _tm.initialize({ 1 }, 4, 1, 0.3, 0.5, 1, 6, 0.1, 0.05, 42);

  Sequence sequence;
  sequence.data.push_back( _patternMachine.get(0) );

  for (int i = 0; i < 4; i++)
    _feedTM(sequence);

  for (int i = 0; i < 2; i++)
    _feedTM(sequence, 10);

}


// ==============================
// Overrides
// ==============================

void TemporalMemoryTutorialTest::setUp()
{
/*
  print("\n"
    "======================================================\n"
    "Test: {0} \n"
    "{1}\n"
    "======================================================\n"
    ).format(self.id(), self.shortDescription());
*/
  _verbosity = 1;
  _tm.initialize({ 6 }, 4, 1, 0.3, 0.5, 1, 6, 0.1, 0.05, 42);

  patternMachine = ConsecutivePatternMachine();
  patternMachine.initialize(6, vector<UInt>{ 1 }, 100, 42);

  _sequenceMachine = SequenceMachine(patternMachine);

//  TemporalMemoryAbstractTest::setUp();

}


void TemporalMemoryTutorialTest::init()
{
  TemporalMemoryAbstractTest::init();
  _tm.initialize({ 6 }, 4, 1, 0.3, 0.5, 1, 6, 0.1, 0.05, 42);

  _patternMachine = utils::ConsecutivePatternMachine();
  _patternMachine.initialize(6, vector<UInt>{ 1 }, 100, 42);
  _sequenceMachine = utils::SequenceMachine(_patternMachine, 42);

  cout << "Initialized new TM with parameters:" << endl;
//  cout << pprint.pformat(_computeTMParams(kwargs.get("overrides")));
  
  _tm.printParameters();
  cout << endl;
}

void TemporalMemoryTutorialTest::_feedTM(Sequence& sequence, int num, bool learn)
{
  _showInput(sequence, learn, num);

  TemporalMemoryAbstractTest::_feedTM(sequence, learn, num);

  cout << _tm.mmPrettyPrintTraces(_tm.mmGetDefaultTraces(2), _tm.mmGetTraceResets()) << endl;

  if (learn)
    cout << _tm.mmPrettyPrintConnections();

}

// ==============================
// Helper functions
// ==============================

void TemporalMemoryTutorialTest::_showInput(Sequence& sequence, bool learn, int num)
{
  string sequenceText = _sequenceMachine.prettyPrintSequence(sequence, _verbosity);
  string learnText = "(learning " + string(learn ? "enabled" : "disabled") + ")";
  string numText = (num <= 1 ? "" : " [" + ::to_string(num) + " times]");
  
  cout << "Feeding sequence " + learnText + numText + ":\n" + sequenceText;
  cout << endl;

}
