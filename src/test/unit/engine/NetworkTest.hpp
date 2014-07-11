#include "catch.hpp"

#include <nta/engine/Network.hpp>
#include <nta/engine/NuPIC.hpp>
#include <nta/engine/Region.hpp>
#include <nta/ntypes/Dimensions.hpp>

using namespace nta;

SCENARIO( "creating a network should auto-initialize NuPIC", "[network]" ) {

  GIVEN( "uninitialized NuPIC" ) {

    // Uninitialize NuPIC since this test checks auto-initialization
    // If shutdown fails, there is probably a problem with another test which 
    // is not cleaning up its networks. 
    if (NuPIC::isInitialized())
    {
      NuPIC::shutdown();
    }

    THEN("NuPIC should be not initialized") {
      REQUIRE(!NuPIC::isInitialized());
    }

    WHEN("creates a network") {
      { // net constructor called
        Network net;

        THEN("NuPIC should be initialized") {
          CHECK(NuPIC::isInitialized());
        }

        WHEN("adds a region to the network") {
          Region *l1 = net.addRegion("level1", "TestNode", "");

          THEN("region name should be as specified") {
            CHECK("level1" == l1->getName());
          }

          AND_THEN("NuPIC should fail to shutdown") {
            // Network still exists, so this should fail. 
            CHECK_THROWS(NuPIC::shutdown());
          }
        }

      } // net destructor called

      WHEN("No network exists and NuPIC initialized") {

        REQUIRE(NuPIC::isInitialized());

        THEN("NuPIC can be shut down") {
          // net destructor has been called so we should be able to shut down NuPIC now
          CHECK_NOTHROW(NuPIC::shutdown());
        }
      }
    }
  }
}

SCENARIO( "a network can manipulate and access regions", "[network]" ) {

  GIVEN("an empty network") {
    Network net;

    WHEN("add a region of invalid node type") {
      THEN("should fail") {
        CHECK_THROWS(net.addRegion("level1", "nonexistent_nodetype", ""));
      }
    }

    WHEN("add a region of valid node type") {
      // Should be able to add a region 
      Region *l1 = net.addRegion("level1", "TestNode", "");

      THEN("the region should belong to the network") {
        CHECK(l1 != NULL);

        CHECK(l1->getNetwork() == &net);
      }

      THEN("the network can't find a region by incorrect names") {

        CHECK_THROWS(net.getRegions().getByName("nosuchregion"));

        // Make sure partial matches don't work
        CHECK_THROWS(net.getRegions().getByName("level"));
      }

      THEN("the network can find the region by the correct name") {
        Region* l1a = net.getRegions().getByName("level1");
        CHECK(l1a == l1);
      }

      THEN("should not be able to add a second region with the same name") {
        CHECK_THROWS(net.addRegion("level1", "TestNode", ""));
      }
    }
  }

}


SCENARIO( "a network can initialize and run only if regions are assigned with dimensions", "[network]" ) {

  GIVEN("an empty network") {

    Network net;

    WHEN("no regions are added") {

      THEN("it can be initialized") {

        net.initialize();

      }
    }

    WHEN("add a region of no dimensions") {
      // Should be able to add a region 
      Region *l1 = net.addRegion("level1", "TestNode", "");

      THEN("it should fail to initialize or run") {

        // Region does not yet have dimensions -- prevents network initialization
        CHECK_THROWS(net.initialize());
        CHECK_THROWS(net.run(1));
      }

      WHEN("assign dimensions to the region") {

        Dimensions d;
        d.push_back(4);
        d.push_back(4);
        
        l1->setDimensions(d);

        THEN("it can initialize and run") {

          // Should succeed since dimensions are now set
          net.initialize();
          net.run(1);

        }

        WHEN("add more regions") {

          THEN("it can run only if regions are assigned with dimensions") {

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
      }
    }

  }
}

SCENARIO( "a network can be modified in various ways", "[network]" ) {
  {
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

}

SCENARIO( "a network can be unlinked in various ways", "[network]" ) {
  { 
    // unlinking tests
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
    CHECK_THROWS(net.removeLink("level1", "level2", "bottomUpOut"))
    
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
          "Cannot remove link [level1.bottomUpOut (region dims: [4 2])  to level2.bottomUpIn (region dims: [2 1])  type: TestFanIn2] because destination region level2 is initialized. Remove the region first.");
      }

      CHECK(caughtException == true);      
    }
           
  }

}


