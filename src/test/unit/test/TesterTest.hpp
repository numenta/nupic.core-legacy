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
Unit tester tester.
*/

#ifndef _H_TESTER_TEST_H
#define _H_TESTER_TEST_H

#include <nupic/test/Tester.hpp>

namespace nupic {
  
  /** 
   * Tests the unit tester interface.
   * 
   */
  class TesterTest : public Tester  {
    
    public:

      /**
       * Constructor
       */
      TesterTest();

      /**
       * Destructor
       */
      virtual ~TesterTest();
      
      /** 
       * Run all appropriate tests.
       */
      virtual void RunTests() override;

      /**
       * Run tests that should fail.
       *
       * @todo In Google Test, there's currently no easy way to verify that these tests
       *  should fail without affecting the test result, so these tests are not 
       *  run but only put together, waiting to see if there's a solution
       */
      void RunTestsShouldFail();
  };
  
} // end namespace nupic

#endif // __TesterTest_hpp__
