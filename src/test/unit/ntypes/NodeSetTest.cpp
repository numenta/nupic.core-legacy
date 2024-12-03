/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Implementation of BasicType test
 */

#include <gtest/gtest.h>
#include <nupic/ntypes/NodeSet.hpp>
#include <nupic/utils/Log.hpp> // Only required because of issue #802

using namespace nupic;

TEST(NodeSetTest, Basic) {
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
