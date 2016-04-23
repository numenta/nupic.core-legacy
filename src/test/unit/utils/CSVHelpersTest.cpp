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
using nupic::Real;
using namespace std;

TEST(CSVHelpers, getLine) 
{
 CSVReader<Real> reader(
   std::string(std::getenv("NUPIC_CORE"))
   + std::string("/src/examples/algorithms/csv.csv"), 3); //demo data
 vector<string> expected1 = {"1", "2", "4"};
 vector<string> expected2 = {"0.2", "1.3", "-1.1"};
 vector<string> res = {};
 UInt i = 0;
 while((res = reader.getLine()).size() > 0) {
  // VectorHelpers::print_vector(res,", ");
  ASSERT_EQ(res[0], expected1[i++]);
 }
};


TEST(CSVHelpers, readColumn)
{
 CSVReader<Real> reader(
   std::string(std::getenv("NUPIC_CORE"))
   + std::string("/src/examples/algorithms/csv.csv"), 3); //demo data
 vector<string> expected1 = {"1", "2", "4"};
 vector<string> expected2 = {"0.2", "1.3", "-1.1"};

 auto col1 = reader.readColumn(0);
 auto col2 = reader.readColumn(1); //FIXME readColumn() needs to use reset_ properly in multiple calls

 for(UInt i = 0; i < col1.size(); i++) {
  ASSERT_EQ(col1[i], expected1[i]); //cast to float TODO use vector_helpers
  ASSERT_EQ(col2[i], expected2[i]);
 }
};
