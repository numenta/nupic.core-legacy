/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of performance tests for Connections
 */

#include <fstream>
#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "ConnectionsPerformanceTest.hpp"

using namespace std;
using namespace nupic;

#define SEED 42

namespace nupic
{

  void ConnectionsPerformanceTest::RunTests()
  {
    srand(SEED);

    testTemporalMemoryUsage();
    testSpatialPoolerUsage();
    testTemporalPoolerUsage();
  }

  /**
   * Tests typical usage of Connections with Temporal Memory.
   */
  void ConnectionsPerformanceTest::testTemporalMemoryUsage()
  {
    clock_t timer = clock();
    UInt numColumns = 2048, w = 40;

    // Initialize

    TemporalMemory tm;
    vector<UInt> columnDim;
    columnDim.push_back(numColumns);
    tm.initialize(columnDim);

    checkpoint(timer, "temporal memory: initialize");

    // Learn

    vector< vector< vector<Cell> > >sequences;
    vector< vector<Cell> >sequence;
    vector<Cell> sdr;

    for (int i = 0; i < 5; i++)
    {
      for (int j = 0; j < 100; j++)
      {
        sdr = randomSDR(numColumns, w);
        sequence.push_back(sdr);
      }

      sequences.push_back(sequence);
    }

    for (int i = 0; i < 5; i++)
    {
      for (auto sequence : sequences)
      {
        for (auto sdr : sequence)
        {
          feedTM(tm, sdr);
          tm.reset();
        }
      }
    }

    checkpoint(timer, "temporal memory: initialize + learn");

    // Test

    for (auto sequence : sequences)
    {
      for (auto sdr : sequence)
      {
        feedTM(tm, sdr, false);
        tm.reset();
      }
    }

    checkpoint(timer, "temporal memory: initialize + learn + test");
  }

  /**
   * Tests typical usage of Connections with Spatial Pooler.
   */
  void ConnectionsPerformanceTest::testSpatialPoolerUsage()
  {
    runSpatialPoolerTest(2048, 2048, 40, 40, "spatial pooler");
  }

  /**
   * Tests typical usage of Connections with Temporal Pooler.
   */
  void ConnectionsPerformanceTest::testTemporalPoolerUsage()
  {
    runSpatialPoolerTest(2048, 16384, 40, 400, "temporal pooler");
  }

  void ConnectionsPerformanceTest::runSpatialPoolerTest(UInt numCells,
                                                        UInt numInputs,
                                                        UInt w,
                                                        UInt numWinners,
                                                        string label)
  {
    clock_t timer = clock();

    Connections connections(numCells, 1, numInputs);
    Cell cell;
    Segment segment;
    vector<Cell> sdr;
    Activity activity;

    // Initialize

    for (UInt c = 0; c < numCells; c++)
    {
      cell = Cell(c);
      segment = connections.createSegment(c);

      for (UInt i = 0; i < numInputs; i++)
      {
        connections.createSynapse(segment, i, (Permanence)rand()/RAND_MAX);
      }
    }

    checkpoint(timer, label + ": initialize");

    // Learn

    vector<Cell> winnerCells;
    SynapseData synapseData;
    Permanence permanence;

    for (int i = 0; i < 500; i++)
    {
      sdr = randomSDR(numInputs, w);
      activity = connections.computeActivity(sdr, 0.5, 0);
      winnerCells = computeSPWinnerCells(numWinners, activity);

      for (Cell winnerCell : winnerCells)
      {
        segment = Segment(0, winnerCell);

        for (Synapse synapse : connections.synapsesForSegment(segment))
        {
          synapseData = connections.dataForSynapse(synapse);
          permanence = synapseData.permanence;

          if (find(sdr.begin(), sdr.end(), synapseData.presynapticCell) != sdr.end())
          {
            permanence += 0.2;
          }
          else
          {
            permanence -= 0.1;
          }

          permanence = max(permanence, (Permanence)0);
          permanence = min(permanence, (Permanence)1);

          // TODO (Question): Remove synapses with 0 permanence?

          connections.updateSynapsePermanence(synapse, permanence);
        }
      }
    }

    checkpoint(timer, label + ": initialize + learn");

    // Test

    for (int i = 0; i < 500; i++)
    {
      sdr = randomSDR(numInputs, w);
      activity = connections.computeActivity(sdr, 0.5, 0);
      winnerCells = computeSPWinnerCells(numWinners, activity);
    }

    checkpoint(timer, label + ": initialize + learn + test");
  }

  void ConnectionsPerformanceTest::checkpoint(clock_t timer, string text)
  {
    float duration = (float)(clock() - timer) / CLOCKS_PER_SEC;
    cout << duration << " in " << text << endl;
  }

  vector<Cell> ConnectionsPerformanceTest::randomSDR(UInt n, UInt w)
  {
    set<UInt> sdrSet = set<UInt>();
    vector<Cell> sdr = vector<Cell>();

    for (UInt i = 0; i < w; i++)
    {
      sdrSet.insert(rand() % (UInt)n);
    }

    for (UInt c : sdrSet)
    {
      sdr.push_back(Cell(c));
    }

    return sdr;
  }

  void ConnectionsPerformanceTest::feedTM(TemporalMemory &tm,
                                          vector<Cell> sdr,
                                          bool learn)
  {
    UInt activeColumns[sdr.size()];

    for (size_t i = 0; i < sdr.size(); i++)
    {
      activeColumns[i] = sdr[i].idx;
    }

    tm.compute(sdr.size(), activeColumns, learn);
  }

  vector<Cell> ConnectionsPerformanceTest::computeSPWinnerCells(UInt numCells,
                                                                Activity& activity)
  {
    vector< pair<Segment, SynapseIdx> > numActiveSynapsesList;
    vector<Cell>winnerCells;

    numActiveSynapsesList.assign(activity.numActiveSynapsesForSegment.begin(),
                                 activity.numActiveSynapsesForSegment.end());

    sort(numActiveSynapsesList.begin(), numActiveSynapsesList.end(),
         [](const pair<Segment, SynapseIdx>& left, const pair<Segment, SynapseIdx>& right)
         {
           return left.second > right.second;
         });

    for (UInt j = 0; j < min(numCells, (UInt)numActiveSynapsesList.size()); j++)
    {
      winnerCells.push_back(numActiveSynapsesList[j].first.cell);
    }

    return winnerCells;
  }

} // end namespace nupic

int main(int argc, char *argv[])
{
  ConnectionsPerformanceTest test = ConnectionsPerformanceTest();
  test.RunTests();
}
