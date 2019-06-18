/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
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
 * Author: David Keeney, July, 2018
 * --------------------------------------------------------------------- */
 
/***********
 *  Includes some routines for testing Region implementations.
 *
 ***********/
#ifndef REGIONTESTUITLITIES_HPP
#define REGIONTESTUITLITIES_HPP
 
#include <iostream>
#include <htm/engine/Region.hpp>

#include "gtest/gtest.h"

using namespace htm;
namespace testing 
{

  void checkGetSetAgainstSpec(std::shared_ptr<Region> region1,
                              size_t expectedSpecCount, 
                              std::set<std::string>& excluded,
                              bool verbose);
  void checkInputOutputsAgainstSpec(std::shared_ptr<Region> region1, bool verbose);

  ::testing::AssertionResult compareParameterArrays(std::shared_ptr<Region> region1,
                                                    std::shared_ptr<Region> region2, 
                                                    std::string parameter, 
                                                    NTA_BasicType type);
  ::testing::AssertionResult captureParameters(std::shared_ptr<Region> 
                                               region1, std::map<std::string, 
                                               std::string>& parameters);
  ::testing::AssertionResult compareParameters(std::shared_ptr<Region> region1, 
                                               std::map<std::string, 
                                               std::string>& parameters);
  ::testing::AssertionResult compareOutputs(std::shared_ptr<Region> region1, 
                                            std::shared_ptr<Region> region2, 
                                            std::string name);
} // namespace testing

#endif //REGIONTESTUITLITIES_HPP
