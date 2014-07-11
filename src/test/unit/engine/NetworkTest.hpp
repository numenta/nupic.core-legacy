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
