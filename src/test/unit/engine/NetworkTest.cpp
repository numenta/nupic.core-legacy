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

#include <nupic/engine/Network.hpp>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Dimensions.hpp>
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


TEST(NetworkTest, AutoInitialization)
{

  // Uninitialize NuPIC since this test checks auto-initialization
  // If shutdown fails, there is probably a problem with another test which 
  // is not cleaning up its networks. 
  if (NuPIC::isInitialized())
    NuPIC::shutdown();

  ASSERT_TRUE(!NuPIC::isInitialized());
  
  // creating a network should auto-initialize NuPIC
  {
    Network net;
    ASSERT_TRUE(NuPIC::isInitialized());
    Region *l1 = net.addRegion("level1", "TestNode", "");
  
    // Use l1 to avoid a compiler warning
    EXPECT_STREQ("level1", l1->getName().c_str());
  
    // Network still exists, so this should fail. 
    EXPECT_THROW(NuPIC::shutdown(), std::exception);
  }
  // net destructor has been called so we should be able to shut down NuPIC now
  NuPIC::shutdown();
}

TEST(NetworkTest, RegionAccess)
{
  Network net;
  EXPECT_THROW( net.addRegion("level1", "nonexistent_nodetype", ""), std::exception );

  // Should be able to add a region 
  Region *l1 = net.addRegion("level1", "TestNode", "");

  ASSERT_TRUE(l1->getNetwork() == &net);

  EXPECT_THROW(net.getRegions().getByName("nosuchregion"), std::exception);

  // Make sure partial matches don't work
  EXPECT_THROW(net.getRegions().getByName("level"), std::exception);

  Region* l1a = net.getRegions().getByName("level1");
  ASSERT_TRUE(l1a == l1);

  // Should not be able to add a second region with the same name
  EXPECT_THROW(net.addRegion("level1", "TestNode", ""), std::exception);

}


TEST(NetworkTest, InitializationBasic)
{
  Network net;
  net.initialize();
}

TEST(NetworkTest, InitializationNoRegions)
{
  Network net;
  Region *l1 = net.addRegion("level1", "TestNode", "");

  // Region does not yet have dimensions -- prevents network initialization
  EXPECT_THROW(net.initialize(), std::exception);
  EXPECT_THROW(net.run(1), std::exception);

  Dimensions d;
  d.push_back(4);
  d.push_back(4);

  l1->setDimensions(d);

  // Should succeed since dimensions are now set
  net.initialize();
  net.run(1);

  Region *l2 = net.addRegion("level2", "TestNode", "");
  EXPECT_THROW(net.initialize(), std::exception);
  EXPECT_THROW(net.run(1), std::exception);

  l2->setDimensions(d);
  net.run(1);

}

TEST(NetworkTest, Modification)
{
  NTA_DEBUG << "Running network modification tests";

  Network net;
  Region *l1 = net.addRegion("level1", "TestNode", "");

  // should have been added at phase0
  std::set<UInt32> phases = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  l1->setDimensions(d);

  net.addRegion("level2", "TestNode", "");

  // should have been added at phase1
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(1) != phases.end());


  net.link("level1", "level2", "TestFanIn2", "");

  const Collection<Region*>& regions = net.getRegions();

  ASSERT_EQ((UInt32)2, regions.getCount());

  // Should succeed since dimensions are now set
  net.initialize();
  net.run(1);
  Region* l2 = regions.getByName("level2");
  Dimensions d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  EXPECT_THROW(net.removeRegion("doesntexist"), std::exception);

  net.removeRegion("level2");
  // net now only contains level1
  ASSERT_EQ((UInt32)1, regions.getCount());
  EXPECT_THROW(regions.getByName("level2"), std::exception);

  // network requires initialization, but this will auto-initialize
  net.run(1);

  ASSERT_TRUE(l1 == regions.getByName("level1"));
  l2 = net.addRegion("level2", "TestNode", "");

  // should have been added at phase1
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(1) != phases.end());

  // network requires initialization, but can't be initialized
  // because level2 is not initialized
  EXPECT_THROW(net.run(1), std::exception);

  net.link("level1", "level2", "TestFanIn2", "");

  // network can be initialized now
  net.run(1);

  ASSERT_EQ((UInt32)2, regions.getCount());
  ASSERT_TRUE(l2 == regions.getByName("level2"));

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);
           
  // add a third region
  Region* l3 = net.addRegion("level3", "TestNode", "");

  // should have been added at phase 2
  phases = net.getPhases("level3");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(2) != phases.end());

  ASSERT_EQ((UInt32)3, regions.getCount());

  // network requires initialization, but can't be initialized
  // because level3 is not initialized
  EXPECT_THROW(net.run(1), std::exception);

  net.link("level2", "level3", "TestFanIn2", "");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)1, d2[0]);
  ASSERT_EQ((UInt32)1, d2[1]);

  // try to remove a region whose outputs are connected
  // this should fail because it would leave the network
  // unrunnable
  EXPECT_THROW(net.removeRegion("level2"), std::exception);
  ASSERT_EQ((UInt32)3, regions.getCount());
  EXPECT_THROW(net.removeRegion("level1"), std::exception);
  ASSERT_EQ((UInt32)3, regions.getCount());

  // this should be ok
  net.removeRegion("level3");
  ASSERT_EQ((UInt32)2, regions.getCount());

  net.removeRegion("level2");
  net.removeRegion("level1");
  ASSERT_EQ((UInt32)0, regions.getCount());

  // build up the network again -- slightly differently with 
  // l1->l2 and l1->l3
  l1 = net.addRegion("level1", "TestNode", "");
  l1->setDimensions(d); 
  net.addRegion("level2", "TestNode", "");
  net.addRegion("level3", "TestNode", "");
  net.link("level1", "level2", "TestFanIn2", "");
  net.link("level1", "level3", "TestFanIn2", "");
  net.initialize();

  // build it up one more time and let the destructor take care of it
  net.removeRegion("level2");
  net.removeRegion("level3");
  net.run(1);

  l2 = net.addRegion("level2", "TestNode", "");
  l3 = net.addRegion("level3", "TestNode", "");
  // try links in reverse order
  net.link("level2", "level3", "TestFanIn2", "");
  net.link("level1", "level2", "TestFanIn2", "");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)1, d2[0]);
  ASSERT_EQ((UInt32)1, d2[1]);

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  // now let the destructor remove everything

}

TEST(NetworkTest, Unlinking)
{ 
  NTA_DEBUG << "Running unlinking tests";
  Network net;
  net.addRegion("level1", "TestNode", "");
  net.addRegion("level2", "TestNode", "");
  Dimensions d;
  d.push_back(4);
  d.push_back(2);
  net.getRegions().getByName("level1")->setDimensions(d);

  net.link("level1", "level2", "TestFanIn2", "");
  ASSERT_TRUE(net.getRegions().getByName("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2", "outputdoesnotexist", "bottomUpIn"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "inputdoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "leveldoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("leveldoesnotexist", "level2"), std::exception);

  // remove the link from the uninitialized network
  net.removeLink("level1", "level2");
  ASSERT_TRUE(net.getRegions().getByName("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2"), std::exception);

  // remove, specifying output/input names
  net.link("level1", "level2", "TestFanIn2", "");
  net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn"), std::exception);

  net.link("level1", "level2", "TestFanIn2", "");
  net.removeLink("level1", "level2", "bottomUpOut");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut"), std::exception);

  // add the link back and initialize (inducing dimensions)
  net.link("level1", "level2", "TestFanIn2", "");
  net.initialize();

  d = net.getRegions().getByName("level2")->getDimensions();
  ASSERT_EQ((UInt32)2, d.size());
  ASSERT_EQ((UInt32)2, d[0]);
  ASSERT_EQ((UInt32)1, d[1]);

  // remove the link. This will fail because we can't 
  // remove a link to an initialized region
  SHOULDFAIL_WITH_MESSAGE(net.removeLink("level1", "level2"), 
                          "Cannot remove link [level1.bottomUpOut (region dims: [4 2])  to level2.bottomUpIn (region dims: [2 1])  type: TestFanIn2] because destination region level2 is initialized. Remove the region first.");

}

typedef std::vector<std::string> callbackData;
callbackData mydata;

void testCallback(Network* net, UInt64 iteration, void* data)
{
  callbackData& thedata = *(static_cast<callbackData*>(data));
  // push region names onto callback data
  const nupic::Collection<Region*>& regions = net->getRegions();
  for (size_t i = 0; i < regions.getCount(); i++)
  {
    thedata.push_back(regions.getByIndex(i).first);
  }
}


std::vector<std::string> computeHistory;
static void recordCompute(const std::string& name)
{
  computeHistory.push_back(name);
}


TEST(NetworkTest, Phases)
{
  Network net;

  // should auto-initialize with max phase
  Region *l1 = net.addRegion("level1", "TestNode", "");
  // Use l1 to avoid a compiler warning
  EXPECT_STREQ("level1", l1->getName().c_str());

  std::set<UInt32> phaseSet = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phaseSet.size());
  ASSERT_TRUE(phaseSet.find(0) != phaseSet.end());


  Region *l2 = net.addRegion("level2", "TestNode", "");
  EXPECT_STREQ("level2", l2->getName().c_str());
  phaseSet = net.getPhases("level2");
  ASSERT_TRUE(phaseSet.size() == 1);
  ASSERT_TRUE(phaseSet.find(1) != phaseSet.end());

  EXPECT_THROW(net.initialize(), std::exception);

  Dimensions d;
  d.push_back(2);
  d.push_back(2);

  l1->setDimensions(d);
  l2->setDimensions(d);
  net.initialize();
  l1->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l2->setParameterUInt64("computeCallback", (UInt64)recordCompute);

  computeHistory.clear();
  net.run(2);
  ASSERT_EQ((UInt32)4, computeHistory.size());
  // use at() to throw an exception if out of range
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level1", computeHistory.at(2).c_str());
  EXPECT_STREQ("level2", computeHistory.at(3).c_str());
  computeHistory.clear();

  phaseSet.clear();
  phaseSet.insert(0);
  phaseSet.insert(2);
  net.setPhases("level1", phaseSet);
  net.run(2);
  ASSERT_EQ((UInt32)6, computeHistory.size());
  if (computeHistory.size() == 6)
  {
    EXPECT_STREQ("level1", computeHistory.at(0).c_str());
    EXPECT_STREQ("level2", computeHistory.at(1).c_str());
    EXPECT_STREQ("level1", computeHistory.at(2).c_str());
    EXPECT_STREQ("level1", computeHistory.at(3).c_str());
    EXPECT_STREQ("level2", computeHistory.at(4).c_str());
    EXPECT_STREQ("level1", computeHistory.at(5).c_str());
  }
  computeHistory.clear();
}

TEST(NetworkTest, MinMaxPhase)
{
  Network n;
  UInt32 minPhase = n.getMinPhase();
  UInt32 maxPhase = n.getMaxPhase();

  ASSERT_EQ((UInt32)0, minPhase);
  ASSERT_EQ((UInt32)0, maxPhase);

  EXPECT_THROW(n.setMinEnabledPhase(1), std::exception);
  EXPECT_THROW(n.setMaxEnabledPhase(1), std::exception);
  Region *l1 = n.addRegion("level1", "TestNode", "");
  Region *l2 = n.addRegion("level2", "TestNode", "");
  Region *l3 = n.addRegion("level3", "TestNode", "");
  Dimensions d;
  d.push_back(1);
  l1->setDimensions(d);
  l2->setDimensions(d);
  l3->setDimensions(d);

  n.initialize();

  l1->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l2->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l3->setParameterUInt64("computeCallback", (UInt64)recordCompute);

  minPhase = n.getMinEnabledPhase();
  maxPhase = n.getMaxEnabledPhase();

  ASSERT_EQ((UInt32)0, minPhase);
  ASSERT_EQ((UInt32)2, maxPhase);

  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)6, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level3", computeHistory.at(2).c_str());
  EXPECT_STREQ("level1", computeHistory.at(3).c_str());
  EXPECT_STREQ("level2", computeHistory.at(4).c_str());
  EXPECT_STREQ("level3", computeHistory.at(5).c_str());


  n.setMinEnabledPhase(0);
  n.setMaxEnabledPhase(1);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)4, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level1", computeHistory.at(2).c_str());
  EXPECT_STREQ("level2", computeHistory.at(3).c_str());

  n.setMinEnabledPhase(1);
  n.setMaxEnabledPhase(1);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)2, computeHistory.size());
  EXPECT_STREQ("level2", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());

  // reset to full network
  n.setMinEnabledPhase(0);
  n.setMaxEnabledPhase(n.getMaxPhase());
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)6, computeHistory.size());
  if (computeHistory.size() == 6)
  {
    EXPECT_STREQ("level1", computeHistory.at(0).c_str());
    EXPECT_STREQ("level2", computeHistory.at(1).c_str());
    EXPECT_STREQ("level3", computeHistory.at(2).c_str());
    EXPECT_STREQ("level1", computeHistory.at(3).c_str());
    EXPECT_STREQ("level2", computeHistory.at(4).c_str());
    EXPECT_STREQ("level3", computeHistory.at(5).c_str());
  }
  // max < min; allowed, but network should not run
  n.setMinEnabledPhase(1);
  n.setMaxEnabledPhase(0);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)0, computeHistory.size());

  // max > network max
  EXPECT_THROW(n.setMaxEnabledPhase(4), std::exception);

  std::set<UInt32> phases;
  phases.insert(4);
  phases.insert(6);
  n.setPhases("level2", phases);
  n.removeRegion("level1");
  // we now have: level2: 4, 6  level3: 2

  minPhase = n.getMinPhase();
  maxPhase = n.getMaxPhase();

  ASSERT_EQ((UInt32)2, minPhase);
  ASSERT_EQ((UInt32)6, maxPhase);

  computeHistory.clear();
  n.run(2);

  ASSERT_EQ((UInt32)6, computeHistory.size());
  EXPECT_STREQ("level3", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level3", computeHistory.at(3).c_str());
  EXPECT_STREQ("level2", computeHistory.at(4).c_str());
  EXPECT_STREQ("level2", computeHistory.at(5).c_str());

}

TEST(NetworkTest, Callback)
{
  Network n;
  n.addRegion("level1", "TestNode", "");
  n.addRegion("level2", "TestNode", "");
  n.addRegion("level3", "TestNode", "");
  Dimensions d;
  d.push_back(1);
  n.getRegions().getByName("level1")->setDimensions(d);
  n.getRegions().getByName("level2")->setDimensions(d);
  n.getRegions().getByName("level3")->setDimensions(d);


  Collection<Network::callbackItem>& callbacks = n.getCallbacks();
  Network::callbackItem callback(testCallback, (void*)(&mydata));
  callbacks.add("Test Callback", callback);

  n.run(2);
  ASSERT_EQ((UInt32)6, mydata.size());
  EXPECT_STREQ("level1", mydata[0].c_str());
  EXPECT_STREQ("level2", mydata[1].c_str());
  EXPECT_STREQ("level3", mydata[2].c_str());
  EXPECT_STREQ("level1", mydata[3].c_str());
  EXPECT_STREQ("level2", mydata[4].c_str());
  EXPECT_STREQ("level3", mydata[5].c_str());

}
