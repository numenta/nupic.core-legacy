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

    REQUIRE(!NuPIC::isInitialized());


    WHEN("creates a network") {
      {
        Network net;

        THEN("NuPIC should be initialized") {
          CHECK(NuPIC::isInitialized());
        }

        WHEN("adds a region to the network") {
          Region *l1 = net.addRegion("level1", "TestNode", "");

          THEN("region name should be as specified") {
            CHECK("level1" == l1->getName());
          }
        }

        AND_THEN("NuPIC should fail to shutdown") {
          // Network still exists, so this should fail. 
          CHECK_THROWS(NuPIC::shutdown());
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