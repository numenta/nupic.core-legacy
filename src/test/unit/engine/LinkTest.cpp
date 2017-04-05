/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of Link test
 */

#include <sstream>

#include <nupic/engine/Input.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/engine/TestNode.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/utils/Log.hpp>
#include "gtest/gtest.h"

using namespace nupic;


TEST(LinkTest, Links)
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


TEST(LinkTest, DelayedLink)
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
    }

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


TEST(LinkTest, DelayedLinkCapnpSerialization)
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


/**
 * Base class for region implementations in this test module. See also
 * L2TestRegion and L4TestRegion.
 */
class TestRegionBase: public RegionImpl
{
public:
  TestRegionBase(const ValueMap& params, Region *region) :
    RegionImpl(region)
  {

    outputElementCount_ = 1;
  }

  TestRegionBase(BundleIO& bundle, Region* region) :
    RegionImpl(region)
  {
  }

  TestRegionBase(capnp::AnyPointer::Reader& proto, Region* region) :
    RegionImpl(region)
  {
  }

  virtual ~TestRegionBase()
  {
  }

  // Serialize state.
  void serialize(BundleIO& bundle) override
  {
  }

  // De-serialize state. Must be called from deserializing constructor
  void deserialize(BundleIO& bundle) override
  {
  }

  // Serialize state with capnp
  using RegionImpl::write;
  void write(capnp::AnyPointer::Builder& anyProto) const override
  {
  }

  // Deserialize state from capnp. Must be called from deserializing
  // constructor.
  using RegionImpl::read;
  void read(capnp::AnyPointer::Reader& anyProto) override
  {
  }

  // Execute a command
  std::string executeCommand(const std::vector<std::string>& args, Int64 index) override
  {
    return "";
  }

  // Per-node size (in elements) of the given output.
  // For per-region outputs, it is the total element count.
  // This method is called only for outputs whose size is not
  // specified in the nodespec.
  size_t getNodeOutputElementCount(const std::string& outputName) override
  {
    if (outputName == "out")
    {
      return outputElementCount_;
    }
    NTA_THROW << "TestRegionBase::getOutputSize -- unknown output " << outputName;
  }

  /**
   * Get a parameter from a write buffer.
   * This method is called only by the typed getParameter*
   * methods in the RegionImpl base class
   *
   * Must be implemented by all subclasses.
   *
   * @param index A node index. (-1) indicates a region-level parameter
   *
   */
  void getParameterFromBuffer(const std::string& name,
                              Int64 index,
                              IWriteBuffer& value) override
  {
  }

  /**
   * Set a parameter from a read buffer.
   * This method is called only by the RegionImpl base class
   * type-specific setParameter* methods
   * Must be implemented by all subclasses.
   *
   * @param index A node index. (-1) indicates a region-level parameter
   */
  void setParameterFromBuffer(const std::string& name,
                            Int64 index,
                            IReadBuffer& value) override
  {
  }

private:
  TestRegionBase();

  // Constructor param specifying per-node output size
  UInt32 outputElementCount_;
};


/*
 * This region's output is computed as: feedForwardIn + lateralIn
 */
class L2TestRegion: public TestRegionBase
{
public:
  L2TestRegion(const ValueMap& params, Region *region) :
    TestRegionBase(params, region)
  {
  }

  L2TestRegion(BundleIO& bundle, Region* region) :
    TestRegionBase(bundle, region)
  {
  }

  L2TestRegion(capnp::AnyPointer::Reader& proto, Region* region) :
    TestRegionBase(proto, region)
  {
  }

  virtual ~L2TestRegion()
  {
  }

  std::string getNodeType()
  {
    return "L2TestRegion";
  }

  // Used by RegionImplFactory to create and cache
  // a nodespec. Ownership is transferred to the caller.
  static Spec* createSpec()
  {
    auto ns = new Spec;

    /* ----- inputs ------- */
    ns->inputs.add(
      "feedForwardIn",
      InputSpec(
        "Feed-forward input for the node",
        NTA_BasicType_UInt64,
        0, // count. omit?
        true, // required?
        false, // isRegionLevel,
        false  // isDefaultInput
        ));

    ns->inputs.add(
      "lateralIn",
      InputSpec(
        "Lateral input for the node",
        NTA_BasicType_UInt64,
        0, // count. omit?
        true, // required?
        false, // isRegionLevel,
        false  // isDefaultInput
        ));

    /* ----- outputs ------ */
    ns->outputs.add(
      "out",
      OutputSpec(
        "Primary output for the node",
        NTA_BasicType_UInt64,
        3, // 1st is output; 2nd is the given feedForwardIn; 3rd is lateralIn
        false, // isRegionLevel
        true // isDefaultOutput
        ));

    return ns;
  }

  /**
   * Inputs/Outputs are made available in initialize()
   * It is always called after the constructor (or load from serialized state)
   */
  void initialize() override
  {
    nodeCount_ = getDimensions().getCount();
    out_ = getOutput("out");
    feedForwardIn_ = getInput("feedForwardIn");
    lateralIn_ = getInput("lateralIn");
  }

  // Compute outputs from inputs and internal state
  void compute() override
  {
    NTA_DEBUG << "> Computing: " << getName() << " <";

    const Array & outputArray = out_->getData();
    NTA_CHECK(outputArray.getCount() == 3);
    NTA_CHECK(outputArray.getType() == NTA_BasicType_UInt64);
    UInt64 *baseOutputBuffer = (UInt64*)outputArray.getBuffer();

    std::vector<UInt64> ffInput;
    feedForwardIn_->getInputForNode(0, ffInput);
    NTA_CHECK(ffInput.size() > 1);

    NTA_DEBUG << getName() << ".compute: ffInput size=" << ffInput.size()
              << "; inputValue=" << ffInput[0];

    std::vector<UInt64> latInput;
    lateralIn_->getInputForNode(0, latInput);
    NTA_CHECK(latInput.size() > 1);

    NTA_DEBUG << getName() << ".compute: latInput size=" << latInput.size()
              << "; inputValue=" << latInput[0];

    // Only the first element of baseOutputBuffer represents region output. We
    // keep track of inputs to the region using the rest of the baseOutputBuffer
    // vector. These inputs are used in the tests.
    baseOutputBuffer[0] = ffInput[0] + latInput[0];
    baseOutputBuffer[1] = ffInput[0];
    baseOutputBuffer[2] = latInput[0];

    NTA_DEBUG << getName() << ".compute: out=" << baseOutputBuffer[0];
  }

private:
  L2TestRegion();

  /* ----- cached info from region ----- */
  size_t nodeCount_;

  // Input/output buffers for the whole region
  const Input *feedForwardIn_;
  const Input *lateralIn_;
  const Output *out_;

};


class L4TestRegion: public TestRegionBase
{
public:
  /*
   * This region's output is computed as: k + feedbackIn
   */
  L4TestRegion(const ValueMap& params, Region *region) :
    TestRegionBase(params, region),
    k_(params.getScalarT<UInt64>("k"))
  {
  }

  L4TestRegion(BundleIO& bundle, Region* region) :
    TestRegionBase(bundle, region),
    k_(0)
  {
  }

  L4TestRegion(capnp::AnyPointer::Reader& proto, Region* region) :
    TestRegionBase(proto, region),
    k_(0)
  {
  }

  virtual ~L4TestRegion()
  {
  }

  std::string getNodeType()
  {
    return "L4TestRegion";
  }

  // Used by RegionImplFactory to create and cache
  // a nodespec. Ownership is transferred to the caller.
  static Spec* createSpec()
  {
    auto ns = new Spec;
    /* ---- parameters ------ */
    ns->parameters.add(
      "k",
      ParameterSpec(
        "Constant k value for output computation", // description
        NTA_BasicType_UInt64,
        1,                         // elementCount
        "",                        // constraints
        "",                        // defaultValue
        ParameterSpec::ReadWriteAccess));

    /* ----- inputs ------- */
    ns->inputs.add(
      "feedbackIn",
      InputSpec(
        "Feedback input for the node",
        NTA_BasicType_UInt64,
        0, // count. omit?
        true, // required?
        false, // isRegionLevel,
        false  // isDefaultInput
        ));

    /* ----- outputs ------ */
    ns->outputs.add(
      "out",
      OutputSpec(
        "Primary output for the node",
        NTA_BasicType_UInt64,
        2, // 2 elements: 1st is output; 2nd is the given feedbackIn value
        false, // isRegionLevel
        true // isDefaultOutput
        ));

    return ns;
  }

  /**
   * Inputs/Outputs are made available in initialize()
   * It is always called after the constructor (or load from serialized state)
   */
  void initialize() override
  {
    nodeCount_ = getDimensions().getCount();
    NTA_CHECK(nodeCount_ == 1);
    out_ = getOutput("out");
    feedbackIn_ = getInput("feedbackIn");
  }

  // Compute outputs from inputs and internal state
  void compute() override
  {
    NTA_DEBUG << "> Computing: " << getName() << " <";

    const Array & outputArray = out_->getData();
    NTA_CHECK(outputArray.getCount() == 2);
    NTA_CHECK(outputArray.getType() == NTA_BasicType_UInt64);
    UInt64 *baseOutputBuffer = (UInt64*)outputArray.getBuffer();

    std::vector<UInt64> nodeInput;
    feedbackIn_->getInputForNode(0, nodeInput);
    NTA_CHECK(nodeInput.size() >= 1);

    NTA_DEBUG << getName() << ".compute: fbInput size=" << nodeInput.size()
              << "; inputValue=" << nodeInput[0];

    // Only the first element of baseOutputBuffer represents region output. We
    // keep track of inputs to the region using the rest of the baseOutputBuffer
    // vector. These inputs are used in the tests.
    baseOutputBuffer[0] = k_ + nodeInput[0];
    baseOutputBuffer[1] = nodeInput[0];

    NTA_DEBUG << getName() << ".compute: out=" << baseOutputBuffer[0];
  }

private:
  L4TestRegion();

  const UInt64 k_;

  /* ----- cached info from region ----- */
  size_t nodeCount_;

  // Input/output buffers for the whole region
  const Input *feedbackIn_;
  const Output *out_;

};


TEST(LinkTest, L2L4WithDelayedLinksAndPhases)
{
  // This test simulates a network with L2 and L4, structured as follows:
  // o R1/R2 ("L4") are in phase 1; R3/R4 ("L2") are in phase 2;
  // o feed-forward links with delay=0 from R1/R2 to R3/R4, respectively;
  // o lateral links with delay=1 between R3 and R4;
  // o feedback links with delay=1 from R3/R4 to R1/R2, respectively

  Network net;

  RegionImplFactory::registerCPPRegion("L4TestRegion",
                                       new RegisteredRegionImpl<L4TestRegion>());
  Region * r1 = net.addRegion("R1", "L4TestRegion", "{\"k\": 1}");
  Region * r2 = net.addRegion("R2", "L4TestRegion", "{\"k\": 5}");
  RegionImplFactory::unregisterCPPRegion("L4TestRegion");

  RegionImplFactory::registerCPPRegion("L2TestRegion",
                                       new RegisteredRegionImpl<L2TestRegion>());
  Region * r3 = net.addRegion("R3", "L2TestRegion", "");
  Region * r4 = net.addRegion("R4", "L2TestRegion", "");
  RegionImplFactory::unregisterCPPRegion("L2TestRegion");

  // NOTE Dimensions must be multiples of 2
  Dimensions d1;
  d1.push_back(1);
  r1->setDimensions(d1);
  r2->setDimensions(d1);
  r3->setDimensions(d1);
  r4->setDimensions(d1);

  /* Set region phases */

  std::set<UInt32> phases;
  phases.insert(1);
  net.setPhases("R1", phases);
  net.setPhases("R2", phases);

  phases.clear();
  phases.insert(2);
  net.setPhases("R3", phases);
  net.setPhases("R4", phases);

  /* Link up the network */

  // R1 output
  net.link(
    "R1", // srcName
    "R3", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",           // srcOutput
    "feedForwardIn", // destInput
    0 //propagationDelay
  );

  // R2 output
  net.link(
    "R2", // srcName
    "R4", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",           // srcOutput
    "feedForwardIn", // destInput
    0 //propagationDelay
  );

  // R3 outputs
  net.link(
    "R3", // srcName
    "R1", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "feedbackIn", // destInput
    1 //propagationDelay
  );

  net.link(
    "R3", // srcName
    "R4", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "lateralIn",  // destInput
    1 //propagationDelay
  );

  // R4 outputs
  net.link(
    "R4", // srcName
    "R2", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "feedbackIn", // destInput
    1 //propagationDelay
  );

  net.link(
    "R4", // srcName
    "R3", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "lateralIn",  // destInput
    1 //propagationDelay
  );

  // Initialize the network
  net.initialize();

  UInt64* r1OutBuf = (UInt64*)(r1->getOutput("out")->getData().getBuffer());
  UInt64* r2OutBuf = (UInt64*)(r2->getOutput("out")->getData().getBuffer());
  UInt64* r3OutBuf = (UInt64*)(r3->getOutput("out")->getData().getBuffer());
  UInt64* r4OutBuf = (UInt64*)(r4->getOutput("out")->getData().getBuffer());

  /* ITERATION #1 */
  net.run(1);

  // Validate R1
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(1u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R2
  ASSERT_EQ(0u, r2OutBuf[1]); // feedbackIn from R4; delay=1
  ASSERT_EQ(5u, r2OutBuf[0]); // out (5 + feedbackIn)

  // Validate R3
  ASSERT_EQ(1u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(0u, r3OutBuf[2]); // lateralIn from R4; delay=1
  ASSERT_EQ(1u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)

  // Validate R4
  ASSERT_EQ(5u, r4OutBuf[1]); // feedForwardIn from R2; delay=0
  ASSERT_EQ(0u, r4OutBuf[2]); // lateralIn from R3; delay=1
  ASSERT_EQ(5u, r4OutBuf[0]); // out (feedForwardIn + lateralIn)


  /* ITERATION #2 */
  net.run(1);

  // Validate R1
  ASSERT_EQ(1u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(2u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R2
  ASSERT_EQ(5u, r2OutBuf[1]);  // feedbackIn from R4; delay=1
  ASSERT_EQ(10u, r2OutBuf[0]); // out (5 + feedbackIn)

  // Validate R3
  ASSERT_EQ(2u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(5u, r3OutBuf[2]); // lateralIn from R4; delay=1
  ASSERT_EQ(7u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)

  // Validate R4
  ASSERT_EQ(10u, r4OutBuf[1]); // feedForwardIn from R2; delay=0
  ASSERT_EQ(1u, r4OutBuf[2]);  // lateralIn from R3; delay=1
  ASSERT_EQ(11u, r4OutBuf[0]); // out (feedForwardIn + lateralIn)


  /* ITERATION #3 */
  net.run(1);

  // Validate R1
  ASSERT_EQ(7u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(8u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R2
  ASSERT_EQ(11u, r2OutBuf[1]); // feedbackIn from R4; delay=1
  ASSERT_EQ(16u, r2OutBuf[0]); // out (5 + feedbackIn)

  // Validate R3
  ASSERT_EQ(8u, r3OutBuf[1]);  // feedForwardIn from R1; delay=0
  ASSERT_EQ(11u, r3OutBuf[2]); // lateralIn from R4; delay=1
  ASSERT_EQ(19u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)

  // Validate R4
  ASSERT_EQ(16u, r4OutBuf[1]); // feedForwardIn from R2; delay=0
  ASSERT_EQ(7u, r4OutBuf[2]);  // lateralIn from R3; delay=1
  ASSERT_EQ(23u, r4OutBuf[0]); // out (feedForwardIn + lateralIn)
}


TEST(LinkTest, L2L4With1ColDelayedLinksAndPhase1OnOffOn)
{
  // Validates processing of incoming delayed and outgoing non-delayed link in
  // the context of a region within a suppressed phase.
  //
  // This test simulates a network with a single L2/L4 column, structured as
  // follows:
  // o R1 ("L4") is in phase 1; R3 ("L2") is in phase 2;
  // o feed-forward link with delay=0 from R1 to R3
  // o lateral link with delay=1 looping back from R3 to itself
  // o feedback link with delay=1 from R3 to R1
  //
  // Running the network:
  // o Run 1 time step with both phases enabled
  // o Disable phase 1 and run two time steps
  // o Enable phase 1 and run two time steps

  Network net;

  RegionImplFactory::registerCPPRegion("L4TestRegion",
                                       new RegisteredRegionImpl<L4TestRegion>());
  Region * r1 = net.addRegion("R1", "L4TestRegion", "{\"k\": 1}");
  RegionImplFactory::unregisterCPPRegion("L4TestRegion");

  RegionImplFactory::registerCPPRegion("L2TestRegion",
                                       new RegisteredRegionImpl<L2TestRegion>());
  Region * r3 = net.addRegion("R3", "L2TestRegion", "");
  RegionImplFactory::unregisterCPPRegion("L2TestRegion");

  // NOTE Dimensions must be multiples of 2
  Dimensions d1;
  d1.push_back(1);
  r1->setDimensions(d1);
  r3->setDimensions(d1);

  /* Set region phases */

  std::set<UInt32> phases;
  phases.insert(1);
  net.setPhases("R1", phases);

  phases.clear();
  phases.insert(2);
  net.setPhases("R3", phases);

  /* Link up the network */

  // R1 output
  net.link(
    "R1", // srcName
    "R3", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",           // srcOutput
    "feedForwardIn", // destInput
    0 //propagationDelay
  );

  // R3 outputs
  net.link(
    "R3", // srcName
    "R1", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "feedbackIn", // destInput
    1 //propagationDelay
  );

  net.link(
    "R3", // srcName
    "R3", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "lateralIn",  // destInput
    1 //propagationDelay
  );


  // Initialize the network
  net.initialize();

  UInt64* r1OutBuf = (UInt64*)(r1->getOutput("out")->getData().getBuffer());
  UInt64* r3OutBuf = (UInt64*)(r3->getOutput("out")->getData().getBuffer());

  /* ITERATION #1 with all phases enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(1u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R3
  ASSERT_EQ(1u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(0u, r3OutBuf[2]); // lateralIn loopback from R3; delay=1
  ASSERT_EQ(1u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)


  /* Disable Phase 1, containing R1 */
  net.setMinEnabledPhase(2);

  /* ITERATION #2 with Phase 1 disabled */
  net.run(1);

  // Validate R1 (it's in a disabled phase, so should be stuck at prior values)
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn
  ASSERT_EQ(1u, r1OutBuf[0]); // out

  // Validate R3
  ASSERT_EQ(1u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(1u, r3OutBuf[2]); // lateralIn loopback from R3; delay=1
  ASSERT_EQ(2u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)


  /* ITERATION #3 with Phase 1 disabled */
  net.run(1);

  // Validate R1 (it's in a disabled phase, so should be stuck at prior values)
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn
  ASSERT_EQ(1u, r1OutBuf[0]); // out

  // Validate R3
  ASSERT_EQ(1u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(2u, r3OutBuf[2]); // lateralIn loopback from R3; delay=1
  ASSERT_EQ(3u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)


  /* Enable Phase 1, containing R1 */
  net.setMinEnabledPhase(1);

  /* ITERATION #4 with all phases enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(3u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(4u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R3
  ASSERT_EQ(4u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(3u, r3OutBuf[2]); // lateralIn loopback from R3; delay=1
  ASSERT_EQ(7u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)

  /* ITERATION #5 with all phases enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(7u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(8u, r1OutBuf[0]); // out (1 + feedbackIn)

  // Validate R3
  ASSERT_EQ(8u, r3OutBuf[1]); // feedForwardIn from R1; delay=0
  ASSERT_EQ(7u, r3OutBuf[2]); // lateralIn loopback from R3; delay=1
  ASSERT_EQ(15u, r3OutBuf[0]); // out (feedForwardIn + lateralIn)

}


TEST(LinkTest, SingleL4RegionWithDelayedLoopbackInAndPhaseOnOffOn)
{
  // Validates processing of outgoing/incoming delayed link in the context of a
  // region within a disabled phase.
  //
  // This test simulates a network with a single L4 region, structured as
  // follows:
  // o R1 ("L4") is in phase 1
  // o Loopback link with delay=1 from R1 to itself
  //
  // Running the network:
  // o Run 1 time step with phase 1 enabled
  // o Disable phase 1 and run two time steps
  // o Enable phase 1 and run two time steps

  Network net;

  RegionImplFactory::registerCPPRegion("L4TestRegion",
                                       new RegisteredRegionImpl<L4TestRegion>());
  Region * r1 = net.addRegion("R1", "L4TestRegion", "{\"k\": 1}");
  RegionImplFactory::unregisterCPPRegion("L4TestRegion");

  // NOTE Dimensions must be multiples of 2
  Dimensions d1;
  d1.push_back(1);
  r1->setDimensions(d1);

  /* Set region phases */

  std::set<UInt32> phases;
  phases.insert(1);
  net.setPhases("R1", phases);

  /* Link up the network */

  // R1 output (loopback)
  net.link(
    "R1", // srcName
    "R1", // destName
    "UniformLink", // linkType
    "",   // linkParams
    "out",        // srcOutput
    "feedbackIn", // destInput
    1 //propagationDelay
  );


  // Initialize the network
  net.initialize();

  UInt64* r1OutBuf = (UInt64*)(r1->getOutput("out")->getData().getBuffer());

  /* ITERATION #1 with phase 1 enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(1u, r1OutBuf[0]); // out (1 + feedbackIn)


  /* Disable Phase 1, containing R1 */
  net.setMaxEnabledPhase(0);

  /* ITERATION #2 with Phase 1 disabled */
  net.run(1);

  // Validate R1 (it's in a disabled phase, so should be stuck at prior values)
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn
  ASSERT_EQ(1u, r1OutBuf[0]); // out

  /* ITERATION #3 with Phase 1 disabled */
  net.run(1);

  // Validate R1 (it's in a disabled phase, so should be stuck at prior values)
  ASSERT_EQ(0u, r1OutBuf[1]); // feedbackIn
  ASSERT_EQ(1u, r1OutBuf[0]); // out


  /* Enable Phase 1, containing R1 */
  net.setMaxEnabledPhase(1);

  /* ITERATION #4 with phase 1 enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(1u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(2u, r1OutBuf[0]); // out (1 + feedbackIn)

  /* ITERATION #5 with phase 1 enabled */
  net.run(1);

  // Validate R1
  ASSERT_EQ(2u, r1OutBuf[1]); // feedbackIn from R3; delay=1
  ASSERT_EQ(3u, r1OutBuf[0]); // out (1 + feedbackIn)
}
