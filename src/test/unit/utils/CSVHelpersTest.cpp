/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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



#include <vector>
#include <algorithm>
#include "nupic/utils/CSVHelpers.hpp"
#include "gtest/gtest.h"

#include "nupic/types/Types.hpp"
//#include "nupic/utils/VectorHelpers.hpp"

using namespace nupic::utils::csv;
using nupic::UInt;
using namespace std;

TEST(CSVHelpers, getLine) 
{
 CSVReader<float> reader(
   std::string(std::getenv("NUPIC_CORE"))
   + std::string("/src/examples/algorithms/csv.csv"), 3); //demo data
 vector<string> expected1 = {"1", "2", "4"};
 vector<string> expected2 = {"0.2", "1.3", "-1.1"};
 vector<string> res = {};
 UInt i = 0;
 while(!reader.eof()) {
  res = reader.getLine();
  // VectorHelpers::print_vector(reader.getLine(),", ");
  ASSERT_EQ(res[0], expected1[i]);
  i++;
  cout << i; 
 }
};

