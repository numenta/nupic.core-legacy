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
 * Implementation of Network test
 */

#include "gtest/gtest.h"

#include <nupic/engine/NetworkFactory.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

#define SHOULDFAIL_WITH_MESSAGE(statement, message) \
  { \
    bool caughtException = false; \
    try { \
      statement; \
    } catch(nupic::LoggingException& e) { \
      caughtException = true; \
      EXPECT_STREQ(message, e.getMessage()) << "statement '" #statement "' should fail with message \"" \
      << message << "\", but failed with message \"" << e.getMessage() << "\""; \
    } catch(...) { \
      FAIL() << "statement '" #statement "' did not generate a logging exception"; \
    } \
    EXPECT_EQ(true, caughtException) << "statement '" #statement "' should fail"; \
  }



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

TEST(NetworkFactory, InvalidYamlPath)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("blah.txt"), "(blah.txt) is not a yaml file.");
}

TEST(NetworkFactory, MissingLinkFieldsFile)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("../../src/test/unit/engine/fixtures/missing-link-fields.yaml"),
   "Invalid network structure file -- bad link (wrong size)");
}

TEST(NetworkFactory, MissingRegionFieldsFile)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("../../src/test/unit/engine/fixtures/missing-region-fields.yaml"),
   "Invalid network structure file -- bad region (wrong size)");
}

TEST(NetworkFactory, ExtraFieldFile)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("../../src/test/unit/engine/fixtures/extra-yaml-fields.yaml"),
   "Invalid network structure file -- contains 3 elements when it should contain 2.");
}

TEST(NetworkFactory, NoRegionsFile)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("../../src/test/unit/engine/fixtures/no-regions.yaml"),
   "Invalid network structure file -- no regions");
}

TEST(NetworkFactory, NoLinksFile)
{
  NetworkFactory nf;
  SHOULDFAIL_WITH_MESSAGE(nf.createNetwork("../../src/test/unit/engine/fixtures/no-links.yaml"),
   "Invalid network structure file -- no links");
}


