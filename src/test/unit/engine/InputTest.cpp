/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of Input test
 */

#include <sstream>

#include <nupic/engine/Input.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/TestNode.hpp>
#include "gtest/gtest.h"

using namespace nupic;

TEST(InputTest, BasicNetworkConstruction)
{
  Network net;
  Region * r1 = net.addRegion("r1", "TestNode", "");
  Region * r2 = net.addRegion("r2", "TestNode", "");

  //Test constructor
  Input x(*r1, NTA_BasicType_Int32, true);
  Input y(*r2, NTA_BasicType_Byte, false);
  EXPECT_THROW(Input i(*r1, (NTA_BasicType)(NTA_BasicType_Last + 1), true),
               std::exception);

  //test getRegion()
  ASSERT_EQ(r1, &(x.getRegion()));
  ASSERT_EQ(r2, &(y.getRegion()));

  //test isRegionLevel()
  ASSERT_TRUE(x.isRegionLevel());
  ASSERT_TRUE(! y.isRegionLevel());

  //test isInitialized()
  ASSERT_TRUE(! x.isInitialized());
  ASSERT_TRUE(! y.isInitialized());

  //test one case of initialize()
  EXPECT_THROW(x.initialize(), std::exception);
  EXPECT_THROW(y.initialize(), std::exception);

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  r1->setDimensions(d1);
  Dimensions d2;
  d2.push_back(4);
  d2.push_back(2);
  r2->setDimensions(d2);
  net.link("r1", "r2", "TestFanIn2", "");

  x.initialize();
  y.initialize();

  //test evaluateLinks()
  //should return 0 because x is initialized
  ASSERT_EQ(0u, x.evaluateLinks());
  //should return 0 because there are no links
  ASSERT_EQ(0u, y.evaluateLinks());

  //test getData()
  const ArrayBase * pa = &(y.getData());
  ASSERT_EQ(0u, pa->getCount());
  Real64* buf = (Real64*)(pa->getBuffer());
  ASSERT_TRUE(buf != nullptr);
}

TEST(InputTest, Links)
{
  Network net;
  Region * region1 = net.addRegion("region1", "TestNode", "");
  Region * region2 = net.addRegion("region2", "TestNode", "");

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  region1->setDimensions(d1);

  net.link("region1", "region2", "TestFanIn2", "");

  //test initialize(), which is called by net.initialize()
  //also test evaluateLinks() which is called here
  net.initialize();
  net.run(1);

  //test that region has correct induced dimensions
  Dimensions d2 = region2->getDimensions();
  ASSERT_EQ(2u, d2.size());
  ASSERT_EQ(4u, d2[0]);
  ASSERT_EQ(2u, d2[1]);

  //test getName() and setName()
  Input * in1 = region1->getInput("bottomUpIn");
  Input * in2 = region2->getInput("bottomUpIn");

  EXPECT_STREQ("bottomUpIn", in1->getName().c_str());
  EXPECT_STREQ("bottomUpIn", in2->getName().c_str());
  in1->setName("uselessName");
  EXPECT_STREQ("uselessName", in1->getName().c_str());
  in1->setName("bottomUpIn");

  //test isInitialized()
  ASSERT_TRUE(in1->isInitialized());
  ASSERT_TRUE(in2->isInitialized());

  //test getLinks()
  std::vector<Link*> links = in2->getLinks();
  ASSERT_EQ(1u, links.size());
  for(auto & link : links) {
    //do something to make sure l[i] is a valid Link*
    ASSERT_TRUE(link != nullptr);
    //should fail because regions are initialized
    EXPECT_THROW(in2->removeLink(link), std::exception);
  }

  //test findLink()
  Link * l1 = in1->findLink("region1", "bottomUpOut");
  ASSERT_TRUE(l1 == nullptr);
  Link * l2 = in2->findLink("region1", "bottomUpOut");
  ASSERT_TRUE(l2 != nullptr);


  //test removeLink(), uninitialize()
  //uninitialize() is called internally from removeLink()
  {
    //can't remove link b/c region1 initialized
    EXPECT_THROW(in2->removeLink(l2), std::exception);
    //can't remove region b/c region1 has links
    EXPECT_THROW(net.removeRegion("region1"), std::exception);
    region1->uninitialize();
    region2->uninitialize();
    EXPECT_THROW(in1->removeLink(l2), std::exception);
    in2->removeLink(l2);
    EXPECT_THROW(in2->removeLink(l2), std::exception);
    //l1 == NULL
    EXPECT_THROW(in1->removeLink(l1), std::exception);
  }
}


TEST(InputTest, DelayedLink)
{
  class MyTestNode : public TestNode
  {
  public:
    MyTestNode(const ValueMap& params, Region *region)
    :  TestNode(params, region)
    {}

    MyTestNode(BundleIO& bundle, Region* region)
    :  TestNode(bundle, region)
    {}

    MyTestNode(capnp::AnyPointer::Reader& proto, Region* region)
    :  TestNode(proto, region)
    {}

    std::string getNodeType()
    {
      return "MyTestNode";
    };

    void compute() override
    {
      // Replace with no-op to preserve output
    }
  };

  RegionImplFactory::registerCPPRegion("MyTestNode",
                                       new RegisteredRegionImpl<MyTestNode>());

  Network net;
  Region * region1 = net.addRegion("region1", "MyTestNode", "");
  Region * region2 = net.addRegion("region2", "TestNode", "");

  RegionImplFactory::unregisterCPPRegion("MyTestNode");

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  region1->setDimensions(d1);

  // NOTE: initial delayed values are set to all 0's
  net.link("region1", "region2", "TestFanIn2", "", "", "",
           2/*propagationDelay*/);

  //test initialize(), which is called by net.initialize()
  net.initialize();

  Input * in1 = region1->getInput("bottomUpIn");
  Input * in2 = region2->getInput("bottomUpIn");
  Output * out1 = region1->getOutput("bottomUpOut");

  //test isInitialized()
  ASSERT_TRUE(in1->isInitialized());
  ASSERT_TRUE(in2->isInitialized());

  //test evaluateLinks(), in1 already initialized
  ASSERT_EQ(0u, in1->evaluateLinks());
  ASSERT_EQ(0u, in2->evaluateLinks());

  //set in2 to all 1's, to detect if net.run fails to update the input.
  {
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 1;
  }

  //set out1 to all 10's
  {
    const ArrayBase * ao1 = &(out1->getData());
    Real64* idata = (Real64*)(ao1->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 10;
  }

  // Check extraction of first delayed value
  {
    // This run should also pick up the 10s
    net.run(1);

    //confirm that in2 is all zeroes
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(0, idata[i]);
  }


  //set out1 to all 100's
  {
    const ArrayBase * ao1 = &(out1->getData());
    Real64* idata = (Real64*)(ao1->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 100;
  }


  // Check extraction of second delayed value
  {
    net.run(1);

    //confirm that in2 is all zeroes
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(0, idata[i]);
  }

  // Check extraction of first "generated" value
  {
    net.run(1);

    //confirm that in2 is now all 10's
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(10, idata[i]);
  }

  // Check extraction of second "generated" value
  {
    net.run(1);

    //confirm that in2 is now all 100's
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(100, idata[i]);
  }

}


TEST(InputTest, DelayedLinkCapnpSerialization)
{
  // Cap'n Proto serialization test of delayed link.

  class MyTestNode : public TestNode
  {
  public:
    MyTestNode(const ValueMap& params, Region *region)
    :  TestNode(params, region)
    {}

    MyTestNode(BundleIO& bundle, Region* region)
    :  TestNode(bundle, region)
    {}

    MyTestNode(capnp::AnyPointer::Reader& proto, Region* region)
    :  TestNode(proto, region)
    {}

    std::string getNodeType()
    {
      return "MyTestNode";
    };

    void compute() override
    {
      // Replace with no-op to preserve output
    }
  };

  RegionImplFactory::registerCPPRegion("MyTestNode",
                                       new RegisteredRegionImpl<MyTestNode>());

  Network net;
  Region * region1 = net.addRegion("region1", "MyTestNode", "");
  Region * region2 = net.addRegion("region2", "TestNode", "");

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  region1->setDimensions(d1);

  // NOTE: initial delayed values are set to all 0's
  net.link("region1", "region2", "TestFanIn2", "", "", "",
           2/*propagationDelay*/);

  net.initialize();

  Input * in1 = region1->getInput("bottomUpIn");
  Input * in2 = region2->getInput("bottomUpIn");
  Output * out1 = region1->getOutput("bottomUpOut");

  //test isInitialized()
  ASSERT_TRUE(in1->isInitialized());
  ASSERT_TRUE(in2->isInitialized());

  //test evaluateLinks(), in1 already initialized
  ASSERT_EQ(0u, in1->evaluateLinks());
  ASSERT_EQ(0u, in2->evaluateLinks());

  //set in2 to all 1's, to detect if net.run fails to update the input.
  {
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 1;
  }

  //set out1 to all 10's
  {
    const ArrayBase * ao1 = &(out1->getData());
    Real64* idata = (Real64*)(ao1->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 10;
  }

  // Check extraction of first delayed value
  {
    // This run should also pick up the 10s
    net.run(1);

    //confirm that in2 is all zeroes
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(0, idata[i]);
  }


  //set out1 to all 100's
  {
    const ArrayBase * ao1 = &(out1->getData());
    Real64* idata = (Real64*)(ao1->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 100;
  }


  // Check extraction of second delayed value
  {
    net.run(1);

    //confirm that in2 is all zeroes
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(0, idata[i]);
  }


  // We should now have two delayed array values: 10's and 100's

  // Serialize the current net
  std::stringstream ss;
  net.write(ss);

  // De-serialize into a new net2
  Network net2;
  net2.read(ss);

  net2.initialize();

  region1 = net2.getRegions().getByName("region1");
  region2 = net2.getRegions().getByName("region2");

  in2 = region2->getInput("bottomUpIn");


  // Check extraction of first "generated" value
  {
    net2.run(1);

    //confirm that in2 is now all 10's
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(10, idata[i]);
  }

  // Check extraction of second "generated" value
  {
    net2.run(1);

    //confirm that in2 is now all 100's
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(100, idata[i]);
  }


  RegionImplFactory::unregisterCPPRegion("MyTestNode");
}


TEST(InputTest, SplitterMap)
{
  Network net;
  Region * region1 = net.addRegion("region1", "TestNode", "");
  Region * region2 = net.addRegion("region2", "TestNode", "");

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  region1->setDimensions(d1);

  //test addLink() indirectly - it is called by Network::link()
  net.link("region1", "region2", "TestFanIn2", "");

  //test initialize(), which is called by net.initialize()
  net.initialize();

  Dimensions d2 = region2->getDimensions();
  Input * in1 = region1->getInput("bottomUpIn");
  Input * in2 = region2->getInput("bottomUpIn");
  Output * out1 = region1->getOutput("bottomUpOut");

  //test isInitialized()
  ASSERT_TRUE(in1->isInitialized());
  ASSERT_TRUE(in2->isInitialized());

  //test evaluateLinks(), in1 already initialized
  ASSERT_EQ(0u, in1->evaluateLinks());
  ASSERT_EQ(0u, in2->evaluateLinks());

  //test prepare
  {
    //set in2 to all zeroes
    const ArrayBase * ai2 = &(in2->getData());
    Real64* idata = (Real64*)(ai2->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 0;

    //set out1 to all 10's
    const ArrayBase * ao1 = &(out1->getData());
    idata = (Real64*)(ao1->getBuffer());
    for (UInt i = 0; i < 64; i++)
      idata[i] = 10;

    //confirm that in2 is still all zeroes
    ai2 = &(in2->getData());
    idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(0, idata[i]);

    in2->prepare();

    //confirm that in2 is now all 10's
    ai2 = &(in2->getData());
    idata = (Real64*)(ai2->getBuffer());
    //only test 4 instead of 64 to cut down on number of tests
    for (UInt i = 0; i < 4; i++)
      ASSERT_EQ(10, idata[i]);
  }

  net.run(2);

  //test getSplitterMap()
  std::vector< std::vector<size_t> > sm;
  sm = in2->getSplitterMap();
  ASSERT_EQ(8u, sm.size());
  ASSERT_EQ(8u, sm[0].size());
  ASSERT_EQ(16u, sm[0][4]);
  ASSERT_EQ(12u, sm[3][0]);
  ASSERT_EQ(31u, sm[3][7]);

  //test getInputForNode()
  std::vector<Real64> input;
  in2->getInputForNode(0, input);
  ASSERT_EQ(1, input[0]);
  ASSERT_EQ(0, input[1]);
  ASSERT_EQ(8, input[5]);
  ASSERT_EQ(9, input[7]);
  in2->getInputForNode(3, input);
  ASSERT_EQ(1, input[0]);
  ASSERT_EQ(6, input[1]);
  ASSERT_EQ(15, input[7]);

  //test getData()
  const ArrayBase * pa = &(in2->getData());
  ASSERT_EQ(64u, pa->getCount());
  Real64* data = (Real64*)(pa->getBuffer());
  ASSERT_EQ(1, data[0]);
  ASSERT_EQ(0, data[1]);
  ASSERT_EQ(1, data[30]);
  ASSERT_EQ(15, data[31]);
  ASSERT_EQ(31, data[63]);
}

TEST(InputTest, LinkTwoRegionsOneInput)
{
  Network net;
  Region * region1 = net.addRegion("region1", "TestNode", "");
  Region * region2 = net.addRegion("region2", "TestNode", "");
  Region * region3 = net.addRegion("region3", "TestNode", "");

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  region1->setDimensions(d1);
  region2->setDimensions(d1);

  net.link("region1", "region3", "TestFanIn2", "");
  net.link("region2", "region3", "TestFanIn2", "");

  net.initialize();

  Dimensions d3 = region3->getDimensions();
  Input * in3 = region3->getInput("bottomUpIn");

  ASSERT_EQ(2u, d3.size());
  ASSERT_EQ(4u, d3[0]);
  ASSERT_EQ(2u, d3[1]);

  net.run(2);

  //test getSplitterMap()
  std::vector< std::vector<size_t> > sm;
  sm = in3->getSplitterMap();
  ASSERT_EQ(8u, sm.size());
  ASSERT_EQ(16u, sm[0].size());
  ASSERT_EQ(16u, sm[0][4]);
  ASSERT_EQ(12u, sm[3][0]);
  ASSERT_EQ(31u, sm[3][7]);

  //test getInputForNode()
  std::vector<Real64> input;
  in3->getInputForNode(0, input);
  ASSERT_EQ(1, input[0]);
  ASSERT_EQ(0, input[1]);
  ASSERT_EQ(8, input[5]);
  ASSERT_EQ(9, input[7]);
  in3->getInputForNode(3, input);
  ASSERT_EQ(1, input[0]);
  ASSERT_EQ(6, input[1]);
  ASSERT_EQ(15, input[7]);

  //test getData()
  const ArrayBase * pa = &(in3->getData());
  ASSERT_EQ(128u, pa->getCount());
  Real64* data = (Real64*)(pa->getBuffer());
  ASSERT_EQ(1, data[0]);
  ASSERT_EQ(0, data[1]);
  ASSERT_EQ(1, data[30]);
  ASSERT_EQ(15, data[31]);
  ASSERT_EQ(31, data[63]);
  ASSERT_EQ(1, data[64]);
  ASSERT_EQ(0, data[65]);
  ASSERT_EQ(1, data[94]);
  ASSERT_EQ(15, data[95]);
  ASSERT_EQ(31, data[127]);

}
