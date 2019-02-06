/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
 
 * Author: David Keeney, July, 2018
 * ---------------------------------------------------------------------
 */
 
/***********
 *  Includes some routines for testing Region implementations.
 *
 ***********/
#ifndef REGIONTESTUITLITIES_HPP
#define REGIONTESTUITLITIES_HPP
 
#include <iostream>
#include <nupic/engine/Region.hpp>
#include <nupic/algorithms/BacktrackingTMCpp.hpp>

#include "gtest/gtest.h"

using namespace nupic;
namespace testing 
{
typedef std::shared_ptr<Region> Region_Ptr_t;

  void checkGetSetAgainstSpec(Region_Ptr_t region1,size_t expectedSpecCount, bool verbose);
  void checkInputOutputsAgainstSpec(Region_Ptr_t region1, bool verbose);

  ::testing::AssertionResult compareParameterArrays(Region_Ptr_t region1,Region_Ptr_t region2, std::string parameter, NTA_BasicType type);
  ::testing::AssertionResult captureParameters(Region_Ptr_t region1, std::map<std::string, std::string>& parameters);
  ::testing::AssertionResult compareParameters(Region_Ptr_t region1, std::map<std::string, std::string>& parameters);
  ::testing::AssertionResult compareOutputs(Region_Ptr_t region1, Region_Ptr_t region2, std::string name);
} // namespace testing

#endif //REGIONTESTUITLITIES_HPP
