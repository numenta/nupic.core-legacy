/*
 * Copyright 2015-2016 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Definitions for ConnectionsPerformanceTest
 */

//----------------------------------------------------------------------

#ifndef NTA_CONNECTIONS_PERFORMANCE_TEST
#define NTA_CONNECTIONS_PERFORMANCE_TEST

#include <vector>

#include <nupic/types/Types.hpp>

namespace nupic {

namespace algorithms {
namespace temporal_memory {
class TemporalMemory;
}

namespace connections {
typedef UInt32 Segment;
}
} // namespace algorithms

class ConnectionsPerformanceTest {
public:
  ConnectionsPerformanceTest() {}
  virtual ~ConnectionsPerformanceTest() {}

  // Run all appropriate tests
  virtual void RunTests();

  void testTemporalMemoryUsage();
  void testLargeTemporalMemoryUsage();
  void testSpatialPoolerUsage();
  void testTemporalPoolerUsage();

private:
  void runTemporalMemoryTest(UInt numColumns, UInt w, int numSequences,
                             int numElements, std::string label);
  void runSpatialPoolerTest(UInt numCells, UInt numInputs, UInt w,
                            UInt numWinners, std::string label);

  void checkpoint(clock_t timer, std::string text);
  std::vector<UInt32> randomSDR(UInt n, UInt w);
  void feedTM(algorithms::temporal_memory::TemporalMemory &tm,
              std::vector<CellIdx> sdr, bool learn = true);
  std::vector<CellIdx>
  computeSPWinnerCells(Connections &connections, UInt numCells,
                       const vector<UInt> &numActiveSynapsesForSegment);

}; // end class ConnectionsPerformanceTest

} // end namespace nupic

#endif // NTA_CONNECTIONS_PERFORMANCE_TEST
