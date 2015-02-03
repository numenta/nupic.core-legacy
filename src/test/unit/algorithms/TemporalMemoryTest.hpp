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
 * Definitions for TemporalMemoryTest
 */

//----------------------------------------------------------------------

#ifndef NTA_TEMPORAL_MEMORY_TEST
#define NTA_TEMPORAL_MEMORY_TEST

#include <nupic/test/Tester.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace nupic::algorithms::temporal_memory;

namespace nupic {

  class TemporalMemoryTest : public Tester
  {
  public:
    TemporalMemoryTest() {}
    virtual ~TemporalMemoryTest() {}

    // Run all appropriate tests
    virtual void RunTests() override;

  private:
    void setup(TemporalMemory& tm, UInt numColumns);

    bool check_vector_eq(UInt arr[], vector<UInt> vec);
    bool check_vector_eq(Real arr[], vector<Real> vec);
    bool check_vector_eq(UInt arr1[], UInt arr2[], UInt n);
    bool check_vector_eq(Real arr1[], Real arr2[], UInt n);
    bool check_vector_eq(vector<UInt> vec1, vector<UInt> vec2);
    void check_spatial_eq(TemporalMemory sp1, TemporalMemory sp2);
    bool almost_eq(Real a, Real b);
    bool findVector(UInt needle[], UInt n, vector<vector<UInt> > haystack);


    void testSaveLoad();
    void testWriteRead();

    void print_vec(UInt arr[], UInt n);
    void print_vec(Real arr[], UInt n);
    void print_vec(vector<UInt> vec);
    void print_vec(vector<Real> vec);

  }; // end class TemporalMemoryTest

} // end namespace nupic

#endif // NTA_TEMPORAL_MEMORY_TEST
