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
#include "nupic/utils/VectorHelpers.hpp"
#include "gtest/gtest.h"
#include "nupic/types/Types.hpp"

namespace testing { 
    
using namespace nupic;
using namespace std;

TEST(VectorHelpersTest, sparseToBinary)
{
  vector<Real32> expected{0.0f,0.0f,1.0f,1.0f,0.0f};
  vector<UInt> v {2u, 3u};
  vector<Real> res = VectorHelpers::sparseToBinary<Real>(v, 5);
  for(size_t i=0; i< res.size(); i++) {
    ASSERT_EQ(res[i], expected[i]);
  }
};

}
