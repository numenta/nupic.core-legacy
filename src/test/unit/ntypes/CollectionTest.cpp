/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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

/** @file
 * Implementation of Collection test
 */

#include <algorithm>
#include <gtest/gtest.h>
#include <htm/ntypes/Collection.hpp>
#include <sstream>

namespace testing { 
    
using namespace htm;

struct CollectionTest : public ::testing::Test {
  struct Item {
    int x;

    Item() : x(-1) {}

    Item(int x) : x(x) {}
    inline bool operator==(const Item &other) const { return x == other.x; };
    inline bool operator!=(const Item &other) const {
      return !operator==(other);
    }
  };
};


TEST_F(CollectionTest, testEmptyCollection) {
  Collection<int> c;
  ASSERT_TRUE(c.getCount() == 0);
  ASSERT_TRUE(c.contains("blah") == false);
  ASSERT_ANY_THROW(c.getByIndex(0));
  ASSERT_ANY_THROW(c.getByName("blah"));
}

TEST_F(CollectionTest, testCollectionWith_1_Item) {
  auto p = new Item(5);
  Collection<Item *> c;
  ASSERT_TRUE(c.contains("x") == false);
  c.add("x", p);
  ASSERT_TRUE(c.contains("x") == true);
  ASSERT_TRUE(c.getCount() == 1);
  ASSERT_TRUE(c.getByIndex(0).second->x == 5);
  ASSERT_TRUE(c.getByName("x")->x == 5);

  ASSERT_ANY_THROW(c.getByIndex(1));
  ASSERT_ANY_THROW(c.getByName("blah"));

  delete p;
}

TEST_F(CollectionTest, testCollectionWith_2_Items) {
  Collection<Item> c;
  c.add("x1", Item(1));
  c.add("x2", Item(2));
  ASSERT_TRUE(c.getCount() == 2);

  Item i1 = c.getByIndex(0).second;
  Item i2 = c.getByIndex(1).second;

  ASSERT_TRUE(i1.x == 1 && i2.x == 2);

  ASSERT_TRUE(c.contains("no such item") == false);
  ASSERT_TRUE(c.contains("x1") == true);
  ASSERT_TRUE(c.contains("x2") == true);
  ASSERT_TRUE(c.getByName("x1").x == 1);
  ASSERT_TRUE(c.getByName("x2").x == 2);

  ASSERT_ANY_THROW(c.getByIndex(2));
  ASSERT_ANY_THROW(c.getByName("blah"));
}

TEST_F(CollectionTest, testCollectionWith_137_Items) {
  Collection<int> c;
  for (int i = 0; i < 137; ++i) {
    std::stringstream ss;
    ss << i;
    c.add(ss.str(), i);
  }

  ASSERT_TRUE(c.getCount() == 137);

  for (int i = 0; i < 137; ++i) {
    ASSERT_TRUE(c.getByIndex(i).second == i);
  }

  ASSERT_ANY_THROW(c.getByIndex(137));
  ASSERT_ANY_THROW(c.getByName("blah"));
}

TEST_F(CollectionTest, testCollectionAddRemove) {
  Collection<int> c;
  c.add("0", 0);
  c.add("1", 1);
  c.add("2", 2);
  // c is now: 0,1,2
  ASSERT_TRUE(c.contains("0"));
  ASSERT_TRUE(c.contains("1"));
  ASSERT_TRUE(c.contains("2"));
  ASSERT_TRUE(!c.contains("3"));

  ASSERT_ANY_THROW(c.add("0", 0));
  ASSERT_ANY_THROW(c.add("1", 1));
  ASSERT_ANY_THROW(c.add("2", 2));

  ASSERT_EQ(0, c.getByName("0"));
  ASSERT_EQ(1, c.getByName("1"));
  ASSERT_EQ(2, c.getByName("2"));

  ASSERT_EQ(0, c.getByIndex(0).second);
  ASSERT_EQ(1, c.getByIndex(1).second);
  ASSERT_EQ(2, c.getByIndex(2).second);

  ASSERT_TRUE(c.getCount() == 3);

  ASSERT_ANY_THROW(c.remove("4"));

  // remove in middle of collection
  c.remove("1");
  // c is now 0, 2
  ASSERT_ANY_THROW(c.remove("1"));

  ASSERT_TRUE(c.getCount() == 2);
  ASSERT_TRUE(c.contains("0"));
  ASSERT_TRUE(!c.contains("1"));
  ASSERT_TRUE(c.contains("2"));

  ASSERT_EQ(0, c.getByIndex(0).second);
  // item "2" has shifted into position 1
  ASSERT_EQ(2, c.getByIndex(1).second);

  // should append to end of collection
  c.add("1", 1);
  // c is now 0, 2, 1
  ASSERT_TRUE(c.getCount() == 3);
  ASSERT_TRUE(c.contains("1"));
  ASSERT_EQ(0, c.getByIndex(0).second);
  ASSERT_EQ(2, c.getByIndex(1).second);
  ASSERT_EQ(1, c.getByIndex(2).second);

  ASSERT_ANY_THROW(c.add("0", 0));
  ASSERT_ANY_THROW(c.add("1", 1));
  ASSERT_ANY_THROW(c.add("2", 2));

  // remove at end of collection
  c.remove("1");
  // c is now 0, 2
  ASSERT_ANY_THROW(c.remove("1"));
  ASSERT_TRUE(c.getCount() == 2);
  ASSERT_EQ(0, c.getByIndex(0).second);
  ASSERT_EQ(2, c.getByIndex(1).second);

  // continue removing until done
  c.remove("0");
  // c is now 2
  ASSERT_ANY_THROW(c.remove("0"));
  ASSERT_TRUE(c.getCount() == 1);
  // "2" shifts to first position
  ASSERT_EQ(2, c.getByIndex(0).second);

  c.remove("2");
  // c is now empty
  ASSERT_TRUE(c.getCount() == 0);
  ASSERT_TRUE(!c.contains("2"));
}

TEST_F(CollectionTest, testCollectionIterator) {
  Collection<int> c;
  c.add("0", 0);
  c.add("1", 1);
  c.add("2", 2);
  // c is now: 0,1,2
  int i = 0;
  for (auto itr :  c) {
  	ASSERT_EQ(i, itr.second);
	i++;
  }
  ASSERT_EQ(3, i);
}
}