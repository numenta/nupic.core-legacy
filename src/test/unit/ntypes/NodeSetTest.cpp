/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

/** @file
 * Implementation of BasicType test
 */

#include <nupic/utils/Log.hpp> // Only required because of issue #802
#include <nupic/ntypes/NodeSet.hpp>
#include <gtest/gtest.h>


using namespace nupic;

TEST(NodeSetTest, Basic)
{
  NodeSet ns(4);
  
  ASSERT_TRUE(ns.begin() == ns.end());
  ns.allOn();
  auto i = ns.begin();
  ASSERT_TRUE(*i == 0);
  ++i;
  ASSERT_TRUE(*i == 1);
  ++i;
  ASSERT_TRUE(*i == 2);
  ++i;
  ASSERT_TRUE(*i == 3);
  ++i;
  ASSERT_TRUE(i == ns.end());
  
  ns.allOff();
  ASSERT_TRUE(ns.begin() == ns.end());
  
  ns.add(1);
  ns.add(3);
  i = ns.begin();
  ASSERT_TRUE(*i == 1);
  ++i;
  ASSERT_TRUE(*i == 3);
  ++i;
  ASSERT_TRUE(i == ns.end());

  ns.add(4);
  i = ns.begin();
  ASSERT_TRUE(*i == 1);
  ++i;
  ASSERT_TRUE(*i == 3);
  ++i;
  ASSERT_TRUE(*i == 4);
  ++i;
  ASSERT_TRUE(i == ns.end());
  
  ASSERT_ANY_THROW(ns.add(5));
  
  ns.remove(3);
  i = ns.begin();
  ASSERT_TRUE(*i == 1);
  ++i;
  ASSERT_TRUE(*i == 4);
  ++i;
  ASSERT_TRUE(i == ns.end());

  // this should have no effect since 3 has already been removed
  ns.remove(3);
  i = ns.begin();
  ASSERT_TRUE(*i == 1);
  ++i;
  ASSERT_TRUE(*i == 4);
  ++i;
  ASSERT_TRUE(i == ns.end());

}
