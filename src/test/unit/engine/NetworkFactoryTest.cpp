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

/** @file
 * Implementation of Network test
 */

#include "gtest/gtest.h"

#include <nupic/engine/NetworkFactory.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

TEST(NetworkFactory, ValidYamlTest)
{
  NetworkFactory nf;
  Network *n;
  n = nf.createNetwork("../../src/test/unit/engine/fixtures/network.yaml");

  const Collection<Region*> regionList = n->getRegions();
  ASSERT_EQ((UInt32)3, regionList.getCount());
  
  // make sure no region specified in the yaml is null.
  Region *l1, *l2, *l3;
  l1 = regionList.getByName("level 1");
  l2 = regionList.getByName("level 2");
  l3 = regionList.getByName("level 3");

  ASSERT_TRUE(l1);
  ASSERT_TRUE(l2);
  ASSERT_TRUE(l3);

  ASSERT_TRUE(l1->getOutput("bottomUpOut"));
  ASSERT_TRUE(l2->getOutput("bottomUpOut"));
  ASSERT_TRUE(l3->getOutput("bottomUpOut"));

  ASSERT_TRUE(l1->getInput("bottomUpIn"));
  ASSERT_TRUE(l2->getInput("bottomUpIn"));
  ASSERT_TRUE(l3->getInput("bottomUpIn"));

}

TEST(NetworkFactory, MissingLinkFieldsFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork("../../src/test/unit/engine/fixtures/missing-link-fields.yaml"),
    std::exception);
}

TEST(NetworkFactory, MissingRegionFieldsFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork("../../src/test/unit/engine/fixtures/missing-region-fields.yaml"),
    std::exception);
}

TEST(NetworkFactory, ExtraFieldFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork("../../src/test/unit/engine/fixtures/extra-yaml-fields.yaml"),
    std::exception);
}

TEST(NetworkFactory, NoRegionsFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork("../../src/test/unit/engine/fixtures/no-regions.yaml"),
    std::exception);
}

TEST(NetworkFactory, NoLinksFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork("../../src/test/unit/engine/fixtures/no-links.yaml"),
    std::exception);
}
