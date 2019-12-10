/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
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
 * Author: David Keeney, Nov, 2019
 * --------------------------------------------------------------------- */

/*---------------------------------------------------------------------
 * This is a test of the ClassifierRegion module.  It does not check the SDRClassifier itself
 * but rather just the plug-in mechanisom to call the SDRClassifier.
 *
 * For those not familiar with GTest:
 *     ASSERT_TRUE(value)   -- Fatal assertion that the value is true.  Test terminates if false.
 *     ASSERT_FALSE(value)   -- Fatal assertion that the value is false. Test terminates if true.
 *     ASSERT_STREQ(str1, str2)   -- Fatal assertion that the strings are equal. Test terminates if false.
 *
 *     EXPECT_TRUE(value)   -- Nonfatal assertion that the value is true.  Test fails but continues if false.
 *     EXPECT_FALSE(value)   -- Nonfatal assertion that the value is false. Test fails but continues if true.
 *     EXPECT_STREQ(str1, str2)   -- Nonfatal assertion that the strings are equal. Test fails but continues if false.
 *
 *     EXPECT_THROW(statement, exception_type) -- nonfatal exception, cought and continues.
 *---------------------------------------------------------------------
 */
#include <htm/engine/Input.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/RegisteredRegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/regions/ClassifierRegion.hpp>
#include <htm/utils/Log.hpp>

#include "RegionTestUtilities.hpp"
#include "gtest/gtest.h"

#define VERBOSE                                                                                                        \
  if (verbose)                                                                                                         \
  std::cerr << "[          ] "
static bool verbose = true; // turn this on to print extra stuff for debugging the test.

const UInt EXPECTED_SPEC_COUNT = 1u; // The number of parameters expected in the ClassifierRegion Spec

using namespace htm;
namespace testing {

// Verify that all parameters are working.
// Assumes that the default value in the Spec is the same as the default
// when creating a region with default constructor.
TEST(ClassifierRegionTest, testSpecAndParameters) {
  // create an RDSERegion region with default parameters
  Network net;

  Spec *ns = ClassifierRegion::createSpec();
  //VERBOSE << *ns << std::endl;

  std::shared_ptr<Region> region1 = net.addRegion("region1", "ClassifierRegion", "{}"); // use default configuration
  std::set<std::string> excluded = {};
  checkGetSetAgainstSpec(region1, EXPECTED_SPEC_COUNT, excluded, verbose);
  checkInputOutputsAgainstSpec(region1, verbose);
}

TEST(ClassifierRegionTest, pluginChecks) {
  VERBOSE << "pluginChecks..." << std::endl;
  Network net;

  size_t regionCntBefore = net.getRegions().size();

  VERBOSE << "  Adding a built-in ClassifierRegion region..." << std::endl;
  std::shared_ptr<Region> region1 = net.addRegion("region1", "ClassifierRegion", "{learn: false}");
  size_t regionCntAfter = net.getRegions().size();
  ASSERT_TRUE(regionCntBefore + 1 == regionCntAfter) << " Expected number of regions to increase by one.  ";
  ASSERT_TRUE(region1->getType() == "ClassifierRegion")
      << " Expected type for region1 to be \"ClassifierRegion\" but type is: " << region1->getType();

  EXPECT_THROW(region1->getOutputData("doesnotexist"), std::exception);
  EXPECT_THROW(region1->getInputData("doesnotexist"), std::exception);

  net.initialize();

  // run() should fail because ClassifierRegion was not passed any data.
  EXPECT_THROW(net.run(1), std::exception);
}

TEST(ClassifierRegionTest, asCategoryDecoder) {
  enum classifier_categories { A, B, C };
  Network net;

  std::shared_ptr<Region> encoder = net.addRegion("encoder", "RDSERegion", "{size: 400, seed: 42, category: true, activeBits: 40}");
  std::shared_ptr<Region> sp = net.addRegion("sp", "SPRegion", "{columnCount: 1000, globalInhibition: true}");
  std::shared_ptr<Region> classifier = net.addRegion("classifier", "ClassifierRegion", "{learn: true}");

  net.link("encoder", "sp", "", "", "encoded", "bottomUpIn");
  net.link("encoder", "classifier", "", "", "bucket", "bucket");
  net.link("sp", "classifier", "", "", "bottomUpOut", "pattern");

  net.initialize();

  classifier_categories cats[] = {A, B, C};
  size_t EPOCH = 1000;

  VERBOSE << "With RDSE encoder, Leaning with enum categories A, B, and C" << std::endl;

  for (size_t i = 0; i < EPOCH; i++) {
    encoder->setParameterReal64("sensedValue", (double)cats[(i % 3)]);
    net.run(1);
  }
  // Turn off learning and see if an input can be identified.
  classifier->setParameterBool("learn", false);
  encoder->setParameterReal64("sensedValue", static_cast<Real64>(A));
  net.run(1);
  const Real64 *titles = reinterpret_cast<const Real64*>(classifier->getOutputData("titles").getBuffer());
  UInt32 predicted = classifier->getOutputData("predicted").item<UInt32>(0);
  EXPECT_EQ(static_cast<UInt32>(titles[predicted]), A) << "expected the category of A";
  const Real64 *pdf = reinterpret_cast<const Real64 *>(classifier->getOutputData("pdf").getBuffer());
  VERBOSE << "Encoded A, Classifier predicted A with a probability of " << pdf[predicted] << std::endl;
  ASSERT_NEAR(pdf[predicted], 0.947, 0.003);

  encoder->setParameterReal64("sensedValue", static_cast<Real64>(B));
  net.run(1);
  titles = reinterpret_cast<const Real64 *>(classifier->getOutputData("titles").getBuffer());
  predicted = classifier->getOutputData("predicted").item<UInt32>(0);
  EXPECT_EQ(static_cast<UInt32>(titles[predicted]), B) << "expected the category of B";
  pdf = reinterpret_cast<const Real64 *>(classifier->getOutputData("pdf").getBuffer());
  VERBOSE << "Encoded B, Classifier predicted B with a probability of " << pdf[predicted] << std::endl;
  ASSERT_NEAR(pdf[predicted], 0.944, 0.003);
}

TEST(ClassifierRegionTest, asRealDecoder) {
  Network net;

  std::shared_ptr<Region> encoder = net.addRegion("encoder", "RDSERegion", "{size: 400, radius: 0.1, seed: 42, activeBits: 40}");
  std::shared_ptr<Region> sp = net.addRegion("sp", "SPRegion", "{columnCount: 1000, globalInhibition: true}");
  std::shared_ptr<Region> classifier = net.addRegion("classifier", "ClassifierRegion", "{learn: true}");

  net.link("encoder", "sp", "", "", "encoded", "bottomUpIn");
  net.link("encoder", "classifier", "", "", "bucket", "bucket");
  net.link("sp", "classifier", "", "", "bottomUpOut", "pattern");

  size_t EPOCH = 5000;
  Random rnd(42);

  VERBOSE << "With RDSE encoder, Leaning with random numbers between -1.0 and +1.0 with radius of 0.1" << std::endl;
  for (int i = 0; i < EPOCH; i++) {
    // learn values between -1.0 and 1.0, with a bucket size of 0.1  (100 buckets)
    // Note, to use a smaller bucket or higher probability, there needs to be a lot more iterations.
    encoder->setParameterReal64("sensedValue", rnd.realRange(-1.0f, +1.0f));
    net.run(1);
  }
  classifier->setParameterBool("learn", false);

  {
    encoder->setParameterReal64("sensedValue", -0.552808);
    net.run(1);
    //VERBOSE << "titles: " << classifier->getOutputData("titles") << std::endl;
    //VERBOSE << "pdf:    " << classifier->getOutputData("pdf") << std::endl;
    const Real64 *titles = reinterpret_cast<const Real64 *>(classifier->getOutputData("titles").getBuffer());
    UInt32 predicted = classifier->getOutputData("predicted").item<UInt32>(0);
    const Real64 *pdf = reinterpret_cast<const Real64 *>(classifier->getOutputData("pdf").getBuffer());
    VERBOSE << "Encoded -0.552808, Classifier predicted " << titles[predicted] << " with a probability of " << pdf[predicted] << std::endl;
    EXPECT_NEAR(titles[predicted], -0.5, 0.01);
    EXPECT_NEAR(pdf[predicted], 0.682351, 0.003);
  }

  {
    encoder->setParameterReal64("sensedValue", +0.830509);
    net.run(1);
    const Real64 *titles = reinterpret_cast<const Real64 *>(classifier->getOutputData("titles").getBuffer());
    UInt32 predicted = classifier->getOutputData("predicted").item<UInt32>(0);
    const Real64 *pdf = reinterpret_cast<const Real64 *>(classifier->getOutputData("pdf").getBuffer());
    VERBOSE << "Encoded +0.830509, Classifier predicted " << titles[predicted] << " with a probability of "
            << pdf[predicted] << std::endl;
    EXPECT_NEAR(titles[predicted], +0.8, 0.1);
    EXPECT_NEAR(pdf[predicted], 0.576886, 0.003);
  }
}

TEST(ClassifierRegionTest, testSerialization) {
  VERBOSE << "testSerialization" << std::endl;
  // NOTE: this test does end-to-end serialize and deserialize with the following modules:
  //   ClassifierRegion, SDRClassifier, Network, Region, Array, RDSERegion, SPRegion, SpatialPooler, Connections,
  //   Random, Links
  //
  Network net1;
  Network net2;

  VERBOSE << "  Setup network" << std::endl;
  std::shared_ptr<Region> encoder1 = net1.addRegion("encoder", "RDSERegion", "{size: 1000, seed: 42, category: true, activeBits: 40}");
  std::shared_ptr<Region> sp1 = net1.addRegion("sp", "SPRegion", "{columnCount: 200, globalInhibition: true}");
  std::shared_ptr<Region> tm1 = net1.addRegion("tm", "TMRegion", "");
  std::shared_ptr<Region> classifier1 = net1.addRegion("classifier", "ClassifierRegion", "{learn: true}");
  net1.link("encoder", "sp", "", "", "encoded", "bottomUpIn");
  net1.link("encoder", "classifier", "", "", "bucket", "bucket");
  net1.link("sp", "tm", "", "", "bottomUpOut", "bottomUpIn");
  net1.link("tm", "classifier", "", "", "bottomUpOut", "pattern");
  net1.initialize();

  VERBOSE << "  Verify that it will run." << std::endl;
  encoder1->setParameterReal64("sensedValue", 5);
  net1.run(1);

  // take a snapshot of parameters in ClassifierRegion at this point
  std::map<std::string, std::string> parameterMap;
  EXPECT_TRUE(captureParameters(classifier1, parameterMap)) << "Capturing parameters before save.";

  Directory::removeTree("TestOutputDir", true);
  std::string filename = "TestOutputDir/ClassifierRegionTest.stream";
  VERBOSE << "  Save it to " << filename << "\n";
  net1.saveToFile(filename, SerializableFormat::JSON);

  VERBOSE << "  Restore from " << filename
          << " into a second network and compare." << std::endl;
  net2.loadFromFile(filename, SerializableFormat::JSON);

  std::shared_ptr<Region> classifier2 = net2.getRegion("classifier");

  ASSERT_TRUE(classifier2->getType() == "ClassifierRegion")
      << " Restored ClassifierRegion region does not have the right type.  Expected ClassifierRegion, found "
      << classifier2->getType();

  EXPECT_TRUE(compareParameters(classifier2, parameterMap))
      << "Conflict when comparing ClassifierRegion parameters after restore with before save.";

  EXPECT_TRUE(net1 == net2) << "Restored Network is not the same as the saved Network.";

  // can we continue with execution?  See if we get any exceptions.
  net2.run(2);

  // cleanup
  Directory::removeTree("TestOutputDir", true);
}

} // namespace testing