/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
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

#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/Connections.hpp>

#include "ConnectionsPerformanceTest.hpp"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::temporal_memory;
using namespace nupic::algorithms::connections;

#define SEED 42

namespace nupic
{

  void ConnectionsPerformanceTest::RunTests()
  {
    srand(SEED);

    testTemporalMemoryUsage();
    testLargeTemporalMemoryUsage();
    testSpatialPoolerUsage();
    testTemporalPoolerUsage();
  }

  /**
   * Tests typical usage of Connections with Temporal Memory.
   */
  void ConnectionsPerformanceTest::testTemporalMemoryUsage()
  {
    runTemporalMemoryTest(2048, 40, 5, 100, "temporal memory");
  }

  /**
   * Tests typical usage of Connections with a large Temporal Memory.
   */
  void ConnectionsPerformanceTest::testLargeTemporalMemoryUsage()
  {
    runTemporalMemoryTest(16384, 328, 3, 40, "temporal memory (large)");
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

  void ConnectionsPerformanceTest::runTemporalMemoryTest(UInt numColumns,
                                                         UInt w,
                                                         int numSequences,
                                                         int numElements,
                                                         string label)
  {
    clock_t timer = clock();

    // Initialize

    TemporalMemory tm;
    vector<UInt> columnDim;
    columnDim.push_back(numColumns);
    tm.initialize(columnDim);

    checkpoint(timer, label + ": initialize");

    // Learn

    vector< vector< vector<CellIdx> > >sequences;
    vector< vector<CellIdx> >sequence;
    vector<CellIdx> sdr;

    for (int i = 0; i < numSequences; i++)
    {
      for (int j = 0; j < numElements; j++)
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

    checkpoint(timer, label + ": initialize + learn");

    // Test

    for (auto sequence : sequences)
    {
      for (auto sdr : sequence)
      {
        feedTM(tm, sdr, false);
        tm.reset();
      }
    }

    checkpoint(timer, label + ": initialize + learn + test");
  }

  void ConnectionsPerformanceTest::runSpatialPoolerTest(UInt numCells,
                                                        UInt numInputs,
                                                        UInt w,
                                                        UInt numWinners,
                                                        string label)
  {
    clock_t timer = clock();

    Connections connections(numCells, 1, numInputs);
    Segment segment;
    vector<CellIdx> sdr;

    // Initialize

    for (UInt c = 0; c < numCells; c++)
    {
      segment = connections.createSegment(c);

      for (UInt i = 0; i < numInputs; i++)
      {
        const Permanence permanence = max((Permanence)0.000001,
                                          (Permanence)rand()/RAND_MAX);
        connections.createSynapse(segment, i, permanence);
      }
    }

    checkpoint(timer, label + ": initialize");

    // Learn

    vector<CellIdx> winnerCells;
    Permanence permanence;

    for (int i = 0; i < 500; i++)
    {
      sdr = randomSDR(numInputs, w);
      vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
      vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
      connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                  numActivePotentialSynapsesForSegment,
                                  sdr, 0.5);
      winnerCells = computeSPWinnerCells(connections, numWinners,
                                         numActiveConnectedSynapsesForSegment);

      for (CellIdx winnerCell : winnerCells)
      {
        segment = connections.getSegment(winnerCell, 0);

        const vector<Synapse>& synapses =
          connections.synapsesForSegment(segment);

        for (SynapseIdx i = 0; i < (SynapseIdx)synapses.size();)
        {
          const Synapse synapse = synapses[i];
          const SynapseData& synapseData = connections.dataForSynapse(synapse);
          permanence = synapseData.permanence;

          if (find(sdr.begin(), sdr.end(), synapseData.presynapticCell) !=
              sdr.end())
          {
            permanence += 0.2;
          }
          else
          {
            permanence -= 0.1;
          }

          permanence = max(permanence, (Permanence)0);
          permanence = min(permanence, (Permanence)1);

          if (permanence == 0)
          {
            connections.destroySynapse(synapse);
            // The synapses list is updated in-place, so don't update `i`.
          }
          else
          {
            connections.updateSynapsePermanence(synapse, permanence);
            i++;
          }
        }
      }
    }

    checkpoint(timer, label + ": initialize + learn");

    // Test

    for (int i = 0; i < 500; i++)
    {
      sdr = randomSDR(numInputs, w);
      vector<UInt32> numActiveConnectedSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
      vector<UInt32> numActivePotentialSynapsesForSegment(
        connections.segmentFlatListLength(), 0);
      connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                  numActivePotentialSynapsesForSegment,
                                  sdr, 0.5);
      winnerCells = computeSPWinnerCells(connections, numWinners,
                                         numActiveConnectedSynapsesForSegment);
    }

    checkpoint(timer, label + ": initialize + learn + test");
  }

  void ConnectionsPerformanceTest::checkpoint(clock_t timer, string text)
  {
    float duration = (float)(clock() - timer) / CLOCKS_PER_SEC;
    cout << duration << " in " << text << endl;
  }

  vector<CellIdx> ConnectionsPerformanceTest::randomSDR(UInt n, UInt w)
  {
    set<UInt> sdrSet = set<UInt>();
    vector<CellIdx> sdr;

    for (UInt i = 0; i < w; i++)
    {
      sdrSet.insert(rand() % (UInt)n);
    }

    for (UInt c : sdrSet)
    {
      sdr.push_back(c);
    }

    return sdr;
  }

  void ConnectionsPerformanceTest::feedTM(TemporalMemory &tm,
                                          vector<CellIdx> sdr,
                                          bool learn)
  {
    vector<UInt> activeColumns;

    for (auto c : sdr)
    {
      activeColumns.push_back(c);
    }

    tm.compute(activeColumns.size(), activeColumns.data(), learn);
  }

  vector<CellIdx> ConnectionsPerformanceTest::computeSPWinnerCells(
    Connections& connections,
    UInt numCells,
    const vector<UInt>& numActiveSynapsesForSegment)
  {
    // Activate every segment, then choose the top few.
    vector<Segment> activeSegments;
    for (Segment segment = 0;
         segment < numActiveSynapsesForSegment.size();
         segment++)
    {
      activeSegments.push_back(segment);
    }

    set<CellIdx> winnerCells;
    std::sort(activeSegments.begin(), activeSegments.end(),
              [&](Segment a, Segment b)
              {
                return
                  numActiveSynapsesForSegment[a] >
                  numActiveSynapsesForSegment[b];
              });

    for (Segment segment : activeSegments)
    {
      winnerCells.insert(connections.cellForSegment(segment));
      if (winnerCells.size() >= numCells)
      {
        break;
      }
    }

    return vector<CellIdx>(winnerCells.begin(), winnerCells.end());
  }

} // end namespace nupic

int main(int argc, char *argv[])
{
  ConnectionsPerformanceTest test = ConnectionsPerformanceTest();
  test.RunTests();
}
