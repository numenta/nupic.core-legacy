/*
 * ----------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have purchased from
 * Numenta, Inc. a separate commercial license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------
 */

#ifndef NTA_NETWORK_TEST_HPP
#define NTA_NETWORK_TEST_HPP

#include <string>
#include <catch.hpp>

#include <nta/engine/Network.hpp>
#include <nta/engine/NuPIC.hpp>
#include <nta/engine/Region.hpp>
#include <nta/ntypes/Dimensions.hpp>
#include <nta/utils/Log.hpp>


using namespace nta;

namespace nta {

  /**
   * Helpers used while testing Network
   */
  namespace network_test_helper {

    typedef std::vector<std::string> callbackData;
    callbackData mydata;

    void testCallback(Network* net, UInt64 iteration, void* data)
    {
      callbackData& thedata = *(static_cast<callbackData*>(data));
      // push region names onto callback data
      const nta::Collection<Region*>& regions = net->getRegions();
      for (size_t i = 0; i < regions.getCount(); i++)
      {
        thedata.push_back(regions.getByIndex(i).first);
      }
    }

    std::vector<std::string> computeHistory;
    void recordCompute(const std::string& name)
    {
      computeHistory.push_back(name);
    }
  }
}

TEST_CASE( "creating a network should auto-initialize NuPIC", "[network]" ) {

  // Uninitialize NuPIC since this test checks auto-initialization
  // If shutdown fails, there is probably a problem with another test which 
  // is not cleaning up its networks. 
  if (NuPIC::isInitialized())
  {
    NuPIC::shutdown();
  }

  // NuPIC should be not initialized
  REQUIRE(!NuPIC::isInitialized());

  { // net constructor called
    Network net;

    // NuPIC should be initialized
    CHECK(NuPIC::isInitialized());

    Region *l1 = net.addRegion("level1", "TestNode", "");

    // region name should be as specified
    CHECK("level1" == l1->getName());

    // Network still exists, so this should fail. 
    CHECK_THROWS(NuPIC::shutdown());

  } // net destructor called

  // NuPIC should still be initialized
  REQUIRE(NuPIC::isInitialized());

  // net destructor has been called so we should be able to shut down NuPIC now
  CHECK_NOTHROW(NuPIC::shutdown());
}

TEST_CASE( "a network can manipulate and access regions", "[network]" ) {

  // an empty network
  Network net;

  // adding a region of invalid node type should fail
  CHECK_THROWS(net.addRegion("level1", "nonexistent_nodetype", ""));

  // adding a region of valid node type should succeed
  Region *l1 = net.addRegion("level1", "TestNode", "");

  // the region should belong to the network
  CHECK(l1 != NULL);
  CHECK(l1->getNetwork() == &net);

  // the network can't find a region by incorrect names
  CHECK_THROWS(net.getRegions().getByName("nosuchregion"));

  // Make sure partial matches don't work
  CHECK_THROWS(net.getRegions().getByName("level"));

  // the network can find the region by the correct name
  Region* l1a = net.getRegions().getByName("level1");
  CHECK(l1a == l1);

  // should not be able to add a second region with the same name
  CHECK_THROWS(net.addRegion("level1", "TestNode", ""));
}

TEST_CASE( "a network can initialize and run only if regions are assigned with dimensions", "[network]" ) {

  Network net;

  // when no regions are added", it can be initialized
  CHECK_NOTHROW(net.initialize(););

  // Should be able to add a region   
  Region *l1 = net.addRegion("level1", "TestNode", "");

  // added a region of no dimensions, it should fail to initialize or run"

  // Region does not yet have dimensions -- prevents network initialization
  CHECK_THROWS(net.initialize());
  CHECK_THROWS(net.run(1));

  // when assign dimensions to the region, it can initialize and run
  {
    Dimensions d;
    d.push_back(4);
    d.push_back(4);
    
    l1->setDimensions(d);

    // Should succeed since dimensions are now set
    net.initialize();
    net.run(1);
  }

  // added more regions, it can run only if regions are assigned with dimensions
  {
    Region *l2 = net.addRegion("level2", "TestNode", "");
    CHECK_THROWS(net.initialize());
    CHECK_THROWS(net.run(1));

    Dimensions d;
    d.push_back(4);
    d.push_back(4);        
    
    l2->setDimensions(d);
    net.run(1);
  }
}

TEST_CASE( "a network can be modified in various ways", "[network]" ) {

  using namespace nta::network_test_helper;
  
  SECTION( "network modification tests" ) {

    Network net;
    Region *l1 = net.addRegion("level1", "TestNode", "");

    // should have been added at phase0
    std::set<UInt32> phases = net.getPhases("level1");
    CHECK((UInt32)1 == phases.size());
    CHECK(phases.find(0) != phases.end());

    Dimensions d;
    d.push_back(4);
    d.push_back(4);
    l1->setDimensions(d);

    net.addRegion("level2", "TestNode", "");

    // should have been added at phase1
    phases = net.getPhases("level2");
    CHECK((UInt32)1 == phases.size());
    CHECK(phases.find(1) != phases.end());

    net.link("level1", "level2", "TestFanIn2", "");
    
    const Collection<Region*>& regions = net.getRegions();

    CHECK((UInt32)2 == regions.getCount());

    // Should succeed since dimensions are now set
    net.initialize();
    net.run(1);
    Region* l2 = regions.getByName("level2");
    Dimensions d2 = l2->getDimensions();
    CHECK((UInt32)2 == d2.size());
    CHECK((UInt32)2 == d2[0]);
    CHECK((UInt32)2 == d2[1]);
    
    CHECK_THROWS(net.removeRegion("doesntexist"));

    net.removeRegion("level2");
    // net now only contains level1
    CHECK((UInt32)1 == regions.getCount());
    CHECK_THROWS(regions.getByName("level2"));
    
    // network requires initialization, but this will auto-initialize
    net.run(1);

    CHECK(l1 == regions.getByName("level1"));
    l2 = net.addRegion("level2", "TestNode", "");

    // should have been added at phase1
    phases = net.getPhases("level2");
    CHECK((UInt32)1 == phases.size());
    CHECK(phases.find(1) != phases.end());

    // network requires initialization, but can't be initialized
    // because level2 is not initialized
    CHECK_THROWS(net.run(1));

    net.link("level1", "level2", "TestFanIn2", "");
    
    // network can be initialized now
    net.run(1);

    CHECK((UInt32)2 == regions.getCount());
    CHECK(l2 == regions.getByName("level2"));

    d2 = l2->getDimensions();
    CHECK((UInt32)2 == d2.size());
    CHECK((UInt32)2 == d2[0]);
    CHECK((UInt32)2 == d2[1]);
               
    // add a third region
    Region* l3 = net.addRegion("level3", "TestNode", "");

    // should have been added at phase 2
    phases = net.getPhases("level3");
    CHECK((UInt32)1 == phases.size());
    CHECK(phases.find(2) != phases.end());

    CHECK((UInt32)3 == regions.getCount());

    // network requires initialization, but can't be initialized
    // because level3 is not initialized
    CHECK_THROWS(net.run(1));

    net.link("level2", "level3", "TestFanIn2", "");
    net.initialize();
    d2 = l3->getDimensions();
    CHECK((UInt32)2 == d2.size());
    CHECK((UInt32)1 == d2[0]);
    CHECK((UInt32)1 == d2[1]);
    
    // try to remove a region whose outputs are connected
    // this should fail because it would leave the network
    // unrunnable
    CHECK_THROWS(net.removeRegion("level2"));
    CHECK((UInt32)3 == regions.getCount());
    CHECK_THROWS(net.removeRegion("level1"));
    CHECK((UInt32)3 == regions.getCount());

    // this should be ok
    net.removeRegion("level3");
    CHECK((UInt32)2 == regions.getCount());

    net.removeRegion("level2");
    net.removeRegion("level1");
    CHECK((UInt32)0 == regions.getCount());
    
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
    CHECK((UInt32)2 == d2.size());
    CHECK((UInt32)1 == d2[0]);
    CHECK((UInt32)1 == d2[1]);

    d2 = l2->getDimensions();
    CHECK((UInt32)2 == d2.size());
    CHECK((UInt32)2 == d2[0]);
    CHECK((UInt32)2 == d2[1]);
    
    // now let the destructor remove everything
    
  }

  SECTION( "network unlinking tests" ) { 

    Network net;
    net.addRegion("level1", "TestNode", "");
    net.addRegion("level2", "TestNode", "");
    Dimensions d;
    d.push_back(4);
    d.push_back(2);
    net.getRegions().getByName("level1")->setDimensions(d);

    net.link("level1", "level2", "TestFanIn2", "");
    CHECK(net.getRegions().getByName("level2")->getDimensions().isUnspecified());

    CHECK_THROWS(net.removeLink("level1", "level2", "outputdoesnotexist", "bottomUpIn"));
    CHECK_THROWS(net.removeLink("level1", "level2", "bottomUpOut", "inputdoesnotexist"));
    CHECK_THROWS(net.removeLink("level1", "leveldoesnotexist"));
    CHECK_THROWS(net.removeLink("leveldoesnotexist", "level2"));

    // remove the link from the uninitialized network
    net.removeLink("level1", "level2");
    CHECK(net.getRegions().getByName("level2")->getDimensions().isUnspecified());

    CHECK_THROWS(net.removeLink("level1", "level2"));

    // remove, specifying output/input names
    net.link("level1", "level2", "TestFanIn2", "");
    net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn");
    CHECK_THROWS(net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn"));

    net.link("level1", "level2", "TestFanIn2", "");
    net.removeLink("level1", "level2", "bottomUpOut");
    CHECK_THROWS(net.removeLink("level1", "level2", "bottomUpOut"));
    
    // add the link back and initialize (inducing dimensions)
    net.link("level1", "level2", "TestFanIn2", "");
    net.initialize();

    d = net.getRegions().getByName("level2")->getDimensions();
    CHECK((UInt32)2 == d.size());
    CHECK((UInt32)2 == d[0]);
    CHECK((UInt32)1 == d[1]);
    
    // remove the link. This will fail because we can't 
    // remove a link to an initialized region
    // 
    
    {
      bool caughtException = false;

      try
      {
        net.removeLink("level1", "level2");
      }
      catch(nta::LoggingException& e)
      {
        caughtException = true;
        CHECK(e.getMessage() == 
          std::string("Cannot remove link [level1.bottomUpOut (region dims: [4 2])  to level2.bottomUpIn (region dims: [2 1])  type: TestFanIn2] because destination region level2 is initialized. Remove the region first."));
      }

      CHECK(caughtException == true);      
    }
           
  }

  SECTION( "network phases tests" ) {

    Network net;
  
    // should auto-initialize with max phase
    Region *l1 = net.addRegion("level1", "TestNode", "");
    // Use l1 to avoid a compiler warning
    CHECK("level1" == l1->getName());

    std::set<UInt32> phaseSet = net.getPhases("level1");
    CHECK((UInt32)1 == phaseSet.size());
    CHECK(phaseSet.find(0) != phaseSet.end());


    Region *l2 = net.addRegion("level2", "TestNode", "");
    CHECK("level2" == l2->getName());
    phaseSet = net.getPhases("level2");
    CHECK(phaseSet.size() == 1);
    CHECK(phaseSet.find(1) != phaseSet.end());

    CHECK_THROWS(net.initialize());
  
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
    CHECK((UInt32)4 == computeHistory.size());
    // use at() to throw an exception if out of range
    CHECK("level1" == computeHistory.at(0));
    CHECK("level2" == computeHistory.at(1));
    CHECK("level1" == computeHistory.at(2));
    CHECK("level2" == computeHistory.at(3));
    computeHistory.clear();

    phaseSet.clear();
    phaseSet.insert(0);
    phaseSet.insert(2);
    net.setPhases("level1", phaseSet);
    net.run(2);
    CHECK((UInt32)6 == computeHistory.size());
    if (computeHistory.size() == 6)
    {
      CHECK("level1" == computeHistory.at(0));
      CHECK("level2" == computeHistory.at(1));
      CHECK("level1" == computeHistory.at(2));
      CHECK("level1" == computeHistory.at(3));
      CHECK("level2" == computeHistory.at(4));
      CHECK("level1" == computeHistory.at(5));
    }
    computeHistory.clear();
  }

  SECTION( "network min/max phase tests" ) {

    Network n;
    UInt32 minPhase = n.getMinPhase();
    UInt32 maxPhase = n.getMaxPhase();

    CHECK((UInt32)0 == minPhase);
    CHECK((UInt32)0 == maxPhase);
    
    CHECK_THROWS(n.setMinEnabledPhase(1));
    CHECK_THROWS(n.setMaxEnabledPhase(1));
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

    CHECK((UInt32)0 == minPhase);
    CHECK((UInt32)2 == maxPhase);

    computeHistory.clear();
    n.run(2);
    CHECK((UInt32)6 == computeHistory.size());
    CHECK("level1" == computeHistory.at(0));
    CHECK("level2" == computeHistory.at(1));
    CHECK("level3" == computeHistory.at(2));
    CHECK("level1" == computeHistory.at(3));
    CHECK("level2" == computeHistory.at(4));
    CHECK("level3" == computeHistory.at(5));
    
    
    n.setMinEnabledPhase(0);
    n.setMaxEnabledPhase(1);
    computeHistory.clear();
    n.run(2);
    CHECK((UInt32)4 == computeHistory.size());
    CHECK("level1" == computeHistory.at(0));
    CHECK("level2" == computeHistory.at(1));
    CHECK("level1" == computeHistory.at(2));
    CHECK("level2" == computeHistory.at(3));

    n.setMinEnabledPhase(1);
    n.setMaxEnabledPhase(1);
    computeHistory.clear();
    n.run(2);
    CHECK((UInt32)2 == computeHistory.size());
    CHECK("level2" == computeHistory.at(0));
    CHECK("level2" == computeHistory.at(1));

    // reset to full network
    n.setMinEnabledPhase(0);
    n.setMaxEnabledPhase(n.getMaxPhase());
    computeHistory.clear();
    n.run(2);
    CHECK((UInt32)6 == computeHistory.size());
    if (computeHistory.size() == 6)
    {
      CHECK("level1" == computeHistory.at(0));
      CHECK("level2" == computeHistory.at(1));
      CHECK("level3" == computeHistory.at(2));
      CHECK("level1" == computeHistory.at(3));
      CHECK("level2" == computeHistory.at(4));
      CHECK("level3" == computeHistory.at(5));
    }
    // max < min; allowed, but network should not run
    n.setMinEnabledPhase(1);
    n.setMaxEnabledPhase(0);
    computeHistory.clear();
    n.run(2);
    CHECK((UInt32)0 == computeHistory.size());

    // max > network max
    CHECK_THROWS(n.setMaxEnabledPhase(4));
    
    std::set<UInt32> phases;
    phases.insert(4);
    phases.insert(6);
    n.setPhases("level2", phases);
    n.removeRegion("level1");
    // we now have: level2: 4, 6  level3: 2
    
    minPhase = n.getMinPhase();
    maxPhase = n.getMaxPhase();

    CHECK((UInt32)2 == minPhase);
    CHECK((UInt32)6 == maxPhase);

    computeHistory.clear();
    n.run(2);

    CHECK((UInt32)6 == computeHistory.size());
    CHECK("level3" == computeHistory.at(0));
    CHECK("level2" == computeHistory.at(1));
    CHECK("level2" == computeHistory.at(2));
    CHECK("level3" == computeHistory.at(3));
    CHECK("level2" == computeHistory.at(4));
    CHECK("level2" == computeHistory.at(5));
    
    
  }

  SECTION( "network callback tests" ) {

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
    CHECK((UInt32)6 == mydata.size());
    CHECK("level1" == mydata[0]);
    CHECK("level2" == mydata[1]);
    CHECK("level3" == mydata[2]);
    CHECK("level1" == mydata[3]);
    CHECK("level2" == mydata[4]);
    CHECK("level3" == mydata[5]);

  }
}


#endif // NTA_NETWORK_TEST_HPP
