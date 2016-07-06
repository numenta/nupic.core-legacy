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
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>


using namespace nupic;

// macro to ease nesting
#define GP Path::getParent

const std::string UNIT_TESTS_EXECUTABLE_PATH = Path::getExecutablePath();
const std::string NUPIC_CORE_PATH =  GP(GP(GP(GP(UNIT_TESTS_EXECUTABLE_PATH))));
const std::string PATH_TO_FIXTURES = Path::join(NUPIC_CORE_PATH, "src/test/unit/engine/fixtures");

TEST(NetworkFactory, ValidYamlTest)
{
  NetworkFactory nf;
  Network n = nf.createNetwork(Path::join(PATH_TO_FIXTURES, "network.yaml"));

  const Collection<Region*> regionList = n.getRegions();
  ASSERT_EQ((UInt32)3, regionList.getCount());

  // make sure no region specified in the yaml is null.
  Region *l1, *l2, *l3;
  l1 = regionList.getByName("level 1");
  l2 = regionList.getByName("level 2");
  l3 = regionList.getByName("level 3");

  l2->removeAllIncomingLinks();

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
  EXPECT_THROW(nf.createNetwork(Path::join(PATH_TO_FIXTURES, "missing-link-fields.yaml")),
    std::exception);
}

TEST(NetworkFactory, MissingRegionFieldsFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork(Path::join(PATH_TO_FIXTURES, "missing-region-fields.yaml")),
    std::exception);
}

TEST(NetworkFactory, ExtraFieldFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork(Path::join(PATH_TO_FIXTURES, "extra-yaml-fields.yaml")),
    std::exception);
}

TEST(NetworkFactory, NoRegionsFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork(Path::join(PATH_TO_FIXTURES, "no-regions.yaml")),
    std::exception);
}

TEST(NetworkFactory, NoLinksFile)
{
  NetworkFactory nf;
  EXPECT_THROW(nf.createNetwork(Path::join(PATH_TO_FIXTURES, "no-links.yaml")),
    std::exception);
}

TEST(NetworkFactory, EmptyRegions)
{
  NetworkFactory nf;
  Network n;
  n = nf.createNetwork(Path::join(PATH_TO_FIXTURES, "empty-regions.yaml"));
  const Collection<Region*> regionList = n.getRegions();
  ASSERT_EQ((UInt32)0, regionList.getCount());
}
