/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * Definitions for ConnectionsTest
 */

//----------------------------------------------------------------------

#ifndef NTA_CONNECTIONS_TEST
#define NTA_CONNECTIONS_TEST

#include <nupic/test/Tester.hpp>
#include <nupic/algorithms/Connections.hpp>

using namespace nupic::algorithms::connections;

namespace nupic
{

  class ConnectionsTest : public Tester
  {
  public:
    ConnectionsTest() {}
    virtual ~ConnectionsTest() {}

    // Run all appropriate tests
    virtual void RunTests() override;

  private:
    void testConstructor();
    void testCreateSegment();
    void testCreateSegmentReuse();
    void testCreateSynapse();
    void testDestroySegment();
    void testDestroySynapse();
    void testUpdateSynapsePermanence();
    void testMostActiveSegmentForCells();
    void testMostActiveSegmentForCellsNone();
    void testLeastRecentlyUsedSegment();
    void testComputeActivity();
    void testActiveSegments();
    void testActiveCells();
    void testNumSegments();
    void testNumSynapses();

    void setupSampleConnections(Connections &connections);
    Activity computeSampleActivity(Connections &connections);

  }; // end class ConnectionsTest

} // end namespace nupic

#endif // NTA_CONNECTIONS_TEST
