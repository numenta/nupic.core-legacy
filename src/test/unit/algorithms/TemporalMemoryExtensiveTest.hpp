/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#ifndef NTA_EXTENSIVETEMPORALMEMORY_TEST
#define NTA_EXTENSIVETEMPORALMEMORY_TEST

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/PatternMachine.hpp>
#include <nupic/utils/SequenceMachine.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include "TemporalMemoryAbstractTest.hpp"

using namespace std;


namespace nupic {

  class TemporalMemoryExtensiveTest : public TemporalMemoryAbstractTest
  {
  public:
    virtual void RunTests();

    void testB1();
    void testB3();
    void testB4();
    void testB5();
    void testB6();
    void testB7();
    void testB8();
    void testB9();
    void testB11();

    void testH1();
    void testH2();
    void testH3();
    void testH4();
    void testH5();
    void testH9();

    virtual void setUp();
    virtual void init();

    void _testTM(Sequence& sequence);

  }; // of class ExtensiveTemporalMemoryTest

}; // of namespace nupic

#endif // NTA_EXTENSIVETEMPORALMEMORY_TEST
