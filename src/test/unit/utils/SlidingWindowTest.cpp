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

#include "gtest/gtest.h"
#include "nupic/utils/SlidingWindow.hpp"

using namespace std;
using namespace nupic;
using namespace nupic::util;


TEST(SlidingWindow, Instance)
{
  SlidingWindow<int>  w{3, "test", 1};
  const std::vector<int> iv{1,2,3};
  const SlidingWindow<int> w2{3, std::begin(iv), std::end(iv)};

    ASSERT_EQ(w.size(), 0);
    ASSERT_EQ(w.ID, "test");
    ASSERT_EQ(w.DEBUG, 1);
    ASSERT_EQ(w2.size(), 3);
    ASSERT_TRUE(w.maxCapacity == w2.maxCapacity ); // ==3
    w.append(4);
    ASSERT_EQ(w.size(), 1);
    ASSERT_EQ(w.getData(), w.getLinearizedData());
    w.append(1);
    ASSERT_EQ(w[1], w2[0]); //==1
    w.append(2);
    ASSERT_NE(w, w2);
    int ret = -1;
    w.append(3, &ret);
    ASSERT_EQ(ret, 4);
    ASSERT_EQ(w.getLinearizedData(), w2.getLinearizedData());
    ASSERT_EQ(w, w2);
    ASSERT_NE(w.getData(), w2.getData()); // linearized data are same, but internal buffer representations are not
}
