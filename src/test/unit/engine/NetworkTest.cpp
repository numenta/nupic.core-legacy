/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Implementation of Network test
 */

#include "gtest/gtest.h"

#include <htm/engine/Network.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/engine/RegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/utils/Log.hpp>

namespace testing {

using namespace htm;

static bool verbose = false;
#define VERBOSE if(verbose) std::cerr << "[          ]"

TEST(NetworkTest, RegionAccess) {
  Network net;
  EXPECT_THROW(net.addRegion("level1", "nonexistent_nodetype", ""), std::exception);

  // Should be able to add a region
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  ASSERT_TRUE(l1->getNetwork() == &net);

  EXPECT_THROW(net.getRegion("nosuchregion"), std::exception);

  // Make sure partial matches don't work
  EXPECT_THROW(net.getRegion("level"), std::exception);

  std::shared_ptr<Region> l1a = net.getRegion("level1");
  ASSERT_TRUE(l1a == l1);

  // Should not be able to add a second region with the same name
  EXPECT_THROW(net.addRegion("level1", "TestNode", ""), std::exception);
}

TEST(NetworkTest, InitializationBasic) {
  Network net;
  net.initialize();
}

TEST(NetworkTest, InitializationNoRegions) {
  Network net;
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  // Region does not yet have dimensions -- prevents network initialization

  Dimensions d;
  d.push_back(4);
  d.push_back(4);

  l1->setDimensions(d);

  // Should succeed since dimensions are now set
  net.initialize();
  net.run(1);

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "");
  l2->setDimensions(d);
  net.run(1);
}

TEST(NetworkTest, Modification) {
  NTA_DEBUG << "Running network modification tests";

  Network net;
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  // should have been added at phase0
  std::set<UInt32> phases = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  l1->setDimensions(d);

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "{dim: [2,2]}");

  // should have been added at phase1
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(1) != phases.end());

  net.link("level1", "level2");

  ASSERT_EQ((UInt32)2, net.getRegions().size());

  // Should succeed since dimensions are set
  net.initialize();
  net.run(1);
  l2 = net.getRegion("level2");
  Dimensions d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  EXPECT_THROW(net.removeRegion("doesntexist"), std::exception);

  net.removeRegion("level2");
  // net now only contains level1
  ASSERT_EQ((UInt32)1, net.getRegions().size()) << "Should be only region 'level1' remaining\n";
  EXPECT_THROW(net.getRegion("level2"), std::exception);

  auto links = net.getLinks();
  ASSERT_TRUE(links.size() == 0) << "Removing the destination region hould have removed the link.";

  ASSERT_TRUE(l1 == net.getRegion("level1"));
  l2 = net.addRegion("level2", "TestNode", "dim: [2,2]");

  // should have been added at phase1
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(1) != phases.end());

  net.link("level1", "level2");

  // network can be initialized now
  net.run(1);

  ASSERT_EQ((UInt32)2, net.getRegions().size());
  ASSERT_TRUE(l2 == net.getRegion("level2"));

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  // add a third region
  std::shared_ptr<Region> l3 = net.addRegion("level3", "TestNode", "{dim: [1,1]}");

  // should have been added at phase 2
  phases = net.getPhases("level3");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(2) != phases.end());

  ASSERT_EQ((UInt32)3, net.getRegions().size());

  net.link("level2", "level3");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)1, d2[0]);
  ASSERT_EQ((UInt32)1, d2[1]);

  // try to remove a region whose outputs are connected
  // this should fail because it would leave the network
  // unrunnable
  EXPECT_THROW(net.removeRegion("level2"), std::exception);
  ASSERT_EQ((UInt32)3, net.getRegions().size());
  EXPECT_THROW(net.removeRegion("level1"), std::exception);
  ASSERT_EQ((UInt32)3, net.getRegions().size());

  // this should be ok
  net.removeRegion("level3");
  ASSERT_EQ((UInt32)2, net.getRegions().size());

  net.removeRegion("level2");
  net.removeRegion("level1");
  ASSERT_EQ((UInt32)0, net.getRegions().size());

  // build up the network again -- slightly differently with
  // l1->l2 and l1->l3
  l1 = net.addRegion("level1", "TestNode", "");
  l1->setDimensions(d);
  net.addRegion("level2", "TestNode", "");
  net.addRegion("level3", "TestNode", "");
  net.link("level1", "level2");
  net.link("level1", "level3");
  net.initialize();

  // build it up one more time and let the destructor take care of it
  net.removeRegion("level2");
  net.removeRegion("level3");
  net.run(1);

  l2 = net.addRegion("level2", "TestNode", "");
  l3 = net.addRegion("level3", "TestNode", "");
  // try links in reverse order
  net.link("level2", "level3");
  net.link("level1", "level2");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)4, d2[0]);
  ASSERT_EQ((UInt32)4, d2[1]);

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)4, d2[0]);
  ASSERT_EQ((UInt32)4, d2[1]);

  // now let the destructor remove everything
}

TEST(NetworkTest, Unlinking) {
  VERBOSE << "Running unlinking tests \n";
  Network net;
  net.addRegion("level1", "TestNode", "");
  net.addRegion("level2", "TestNode", "");
  Dimensions d;
  d.push_back(4);
  d.push_back(2);
  net.getRegion("level1")->setDimensions(d);

  net.link("level1", "level2");
  ASSERT_TRUE(net.getRegion("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2", "outputdoesnotexist", "bottomUpIn"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "inputdoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "leveldoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("leveldoesnotexist", "level2"), std::exception);

  // remove the link from the uninitialized network
  net.removeLink("level1", "level2");
  ASSERT_TRUE(net.getRegion("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2"), std::exception);

  // remove, specifying output/input names
  net.link("level1", "level2");
  net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn"), std::exception);

  net.link("level1", "level2");
  net.removeLink("level1", "level2", "bottomUpOut");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut"), std::exception);

  // add the link back and initialize (inducing dimensions)
  net.link("level1", "level2");
  net.initialize();

  d = net.getRegion("level2")->getDimensions();
  ASSERT_EQ((UInt32)2, d.size());
  ASSERT_EQ((UInt32)4, d[0]);
  ASSERT_EQ((UInt32)2, d[1]);

  // remove the link. This will fail because we can't
  // remove a link to an initialized region
  EXPECT_THROW(net.removeLink("level1", "level2"), std::exception)
      << "Cannot remove link [level1.bottomUpOut (region dims: [4 2])  to "
         "level2.bottomUpIn (region dims: [2 1])  type: TestFanIn2] because "
         "destination region level2 is initialized. Remove the region first.";
}

typedef std::vector<std::string> callbackData;
callbackData mydata;

void testCallback(Network *net, UInt64 iteration, void *data) {
  callbackData &thedata = *(static_cast<callbackData *>(data));
  // push region names onto callback data
  const Collection<std::shared_ptr<Region>> &regions = net->getRegions();
  for (auto iter = regions.cbegin(); iter != regions.cend(); ++iter) {
    thedata.push_back(iter->first);
  }
}

std::vector<std::string> computeHistory;
static void recordCompute(const std::string &name) { computeHistory.push_back(name); }

TEST(NetworkTest, Phases) {
  Network net;

  // should auto-initialize with max phase
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");
  // Use l1 to avoid a compiler warning
  EXPECT_STREQ("level1", l1->getName().c_str());

  std::set<UInt32> phaseSet = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phaseSet.size());
  ASSERT_TRUE(phaseSet.find(0) != phaseSet.end());

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "");
  EXPECT_STREQ("level2", l2->getName().c_str());
  phaseSet = net.getPhases("level2");
  ASSERT_TRUE(phaseSet.size() == 1);
  ASSERT_TRUE(phaseSet.find(1) != phaseSet.end());

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
  if (computeHistory.size() == 6) {
    EXPECT_STREQ("level1", computeHistory.at(0).c_str());
    EXPECT_STREQ("level2", computeHistory.at(1).c_str());
    EXPECT_STREQ("level1", computeHistory.at(2).c_str());
    EXPECT_STREQ("level1", computeHistory.at(3).c_str());
    EXPECT_STREQ("level2", computeHistory.at(4).c_str());
    EXPECT_STREQ("level1", computeHistory.at(5).c_str());
  }
  computeHistory.clear();
}

TEST(NetworkTest, MinMaxPhase) {
  Network n;
  UInt32 minPhase = n.getMinPhase();
  UInt32 maxPhase = n.getMaxPhase();

  ASSERT_EQ((UInt32)0, minPhase);
  ASSERT_EQ((UInt32)0, maxPhase);

  EXPECT_THROW(n.setMinEnabledPhase(1), std::exception);
  EXPECT_THROW(n.setMaxEnabledPhase(1), std::exception);
  std::shared_ptr<Region> l1 = n.addRegion("level1", "TestNode", "");
  std::shared_ptr<Region> l2 = n.addRegion("level2", "TestNode", "");
  std::shared_ptr<Region> l3 = n.addRegion("level3", "TestNode", "");
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
  if (computeHistory.size() == 6) {
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

TEST(NetworkTest, Callback) {
  Network n;
  n.addRegion("level1", "TestNode", "");
  n.addRegion("level2", "TestNode", "");
  n.addRegion("level3", "TestNode", "");
  Dimensions d;
  d.push_back(1);
  n.getRegion("level1")->setDimensions(d);
  n.getRegion("level2")->setDimensions(d);
  n.getRegion("level3")->setDimensions(d);

  Collection<Network::callbackItem> &callbacks = n.getCallbacks();
  Network::callbackItem callback(testCallback, (void *)(&mydata));
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

/**
 * Test operator '=='
 */
TEST(NetworkTest, testEqualsOperator) {
  Network n1;
  Network n2;
  ASSERT_TRUE(n1 == n2);
  Dimensions d;
  d.push_back(4);
  d.push_back(4);

  auto l1 = n1.addRegion("level1", "TestNode", "");
  ASSERT_TRUE(n1 != n2);
  auto l2 = n2.addRegion("level1", "TestNode", "");
  ASSERT_TRUE(n1 == n2);
  l1->setDimensions(d);
  ASSERT_TRUE(n1 != n2);
  l2->setDimensions(d);
  ASSERT_TRUE(n1 == n2);

  n1.addRegion("level2", "TestNode", "");
  ASSERT_TRUE(n1 != n2);
  n2.addRegion("level2", "TestNode", "");
  ASSERT_TRUE(n1 == n2);

  n1.link("level1", "level2");
  ASSERT_TRUE(n1 != n2);
  n2.link("level1", "level2");
  ASSERT_TRUE(n1 == n2);

  n1.run(1);
  ASSERT_TRUE(n1 != n2);
  n2.run(1);
  ASSERT_TRUE(n1 == n2);
}
} // namespace testing

namespace htm {
// Note: this sort-of mimics the test in network_test.py "testNetworkPickle"
class LinkRegion : public RegionImpl {
public:
  LinkRegion(const ValueMap &params, Region *region) : RegionImpl(region) { param = 52; }
  LinkRegion(ArWrapper &wrapper, Region *region) : RegionImpl(region) { cereal_adapter_load(wrapper);}

  void initialize() override {}
  void compute() override {
    // This will pass its inputs on to the outputs.
    Array &input_data = getInput("inputs")->getData();
    Array &output_data = getOutput("outputs")->getData();
    input_data.convertInto(output_data);
  }
  size_t getNodeOutputElementCount(const std::string &name) const override { return 5; }

  std::string executeCommand(const std::vector<std::string> &args, Int64 index) override {
    if (args[0] == "HelloWorld" && args.size() == 3)
      return "Hello World says: arg1=" + args[1] + " arg2=" + args[2];
    return "";
  }

  // Include the required code for serialization.
  CerealAdapter;
  template <class Archive> void save_ar(Archive &ar) const {
    ar(cereal::make_nvp("param", param));
  }
  template <class Archive> void load_ar(Archive &ar) {
    ar(cereal::make_nvp("param", param));
  }

  static Spec *createSpec() {
    auto ns = new Spec;
    ns->description = "LinkRegion. Used as a plain simple plugin Region for unit tests only. "
                      "This is not useful for any real applicaton.";
    /* ----- inputs ------- */
    ns->inputs.add("UInt32", InputSpec("UInt32 Data",
                                       NTA_BasicType_UInt32, // type
                                       0,                    // count
                                       false,                // required
                                       true,                 // isRegionLevel,
                                       true                  // isDefaultInput
                                       ));
    ns->inputs.add("Real32", InputSpec("Real32 Data",
                                       NTA_BasicType_Real32, // type
                                       0,                    // count
                                       false,                // required
                                       true,                 // isRegionLevel,
                                       false                 // isDefaultInput
                                       ));

    /* ----- outputs ------ */
    ns->outputs.add("UInt32", OutputSpec("UInt32 Data",
                                         NTA_BasicType_UInt32, // type
                                         0,                    // count is dynamic
                                         true,                 // isRegionLevel
                                         true                  // isDefaultOutput
                                         ));
    ns->outputs.add("Real32", OutputSpec("UInt32 Data",
                                         NTA_BasicType_Real32, // type
                                         0,                    // count is dynamic
                                         true,                 // isRegionLevel
                                         false                 // isDefaultOutput
                                         ));
    /* ---- executeCommand ---- */
    ns->commands.add("HelloWorld",  CommandSpec("Hello world command"));

    return ns;
  }

  bool operator==(const RegionImpl &other) const override { return ((LinkRegion&)other).param == param;}
  inline bool operator!=(const LinkRegion &other) const { return !operator==(other); }

private:
  int param;
};

} // namespace htm

namespace testing {
TEST(NetworkTest, SaveRestore) {
  // Note: this sort-of mimics test in network_test.py "testNetworkPickle"
  Network network;
  network.registerRegion("LinkRegion", new RegisteredRegionImplCpp<LinkRegion>());
  auto r_from = network.addRegion("from", "LinkRegion", "");
  auto r_to = network.addRegion("to", "LinkRegion", "");
  size_t cnt = r_from->getNodeOutputElementCount("from");
  ASSERT_EQ(5u, cnt);

  network.link("from", "to", "", "", "UInt32", "UInt32");
  network.link("from", "to", "", "", "Real32", "Real32");
  network.link("from", "to", "", "", "Real32", "UInt32");
  network.link("from", "to", "", "", "UInt32", "Real32");
  network.initialize();

  std::stringstream ss;
  network.save(ss);

  Network network2;
  network2.load(ss);

  std::string s1 = network.getRegion("to")->executeCommand({"HelloWorld", "26", "64"});
  std::string s2 = network2.getRegion("to")->executeCommand({"HelloWorld", "26", "64"});
  ASSERT_STREQ(s1.c_str(), "Hello World says: arg1=26 arg2=64");
  ASSERT_STREQ(s1.c_str(), s2.c_str());
}

} // namespace testing
