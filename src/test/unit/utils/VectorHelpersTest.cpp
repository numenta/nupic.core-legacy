/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */



#include <vector>
#include <algorithm>
#include "htm/utils/VectorHelpers.hpp"
#include "gtest/gtest.h"
#include "htm/types/Types.hpp"

namespace testing { 
    
using namespace htm;
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
