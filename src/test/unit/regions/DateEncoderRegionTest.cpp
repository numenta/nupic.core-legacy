/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, Numenta, Inc.
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
 * Author: David Keeney, Jan, 2020
 * --------------------------------------------------------------------- */

 
/*---------------------------------------------------------------------
  * This is a test of the ScalarSensor module.  It does not check the ScalarEncoder itself
  * but rather just the plug-in mechanisom to call the ScalarEncoder.
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
#include <htm/regions/DateEncoderRegion.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/RegisteredRegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/utils/Log.hpp>
#include <htm/types/Sdr.hpp>

#include "gtest/gtest.h"
#include "RegionTestUtilities.hpp"

#define VERBOSE if(verbose)std::cerr << "[          ] "
static bool verbose = false;  // turn this on to print extra stuff for debugging the test.

const UInt EXPECTED_SPEC_COUNT =  15u;  // The number of parameters expected in the DateEncoderRegion Spec

using namespace htm;
namespace testing 
{

  // Verify that all parameters are working.
  // Assumes that the default value in the Spec is the same as the default 
  // when creating a region with default constructor.
TEST(DateEncoderRegionTest, testSpecAndParameters)
  {
    // create an ScalarSensor region with default parameters
    Network net;

    Spec* ns = DateEncoderRegion::createSpec();
    VERBOSE << *ns << std::endl;

    std::shared_ptr<Region> region1 = net.addRegion("region1", "DateEncoderRegion", "season_width: 5"); 
    std::set<std::string> excluded = {"season_width"};
    checkGetSetAgainstSpec(region1, EXPECTED_SPEC_COUNT, excluded, verbose);
    checkInputOutputsAgainstSpec(region1, verbose);
  }


	TEST(DateEncoderRegionTest, initialization_with_builtin_impl)
	{
	  VERBOSE << "Creating network..." << std::endl;
	  Network net;

	  size_t regionCntBefore = net.getRegions().size();

	  VERBOSE << "Adding a built-in DateEncoderRegion region..." << std::endl;
          std::shared_ptr<Region> region1 = net.addRegion("region1", "DateEncoderRegion", "{season_width: 5}");
	  size_t regionCntAfter = net.getRegions().size();
	  ASSERT_TRUE(regionCntBefore + 1 == regionCntAfter) << " Expected number of regions to increase by one.  ";
	  ASSERT_TRUE(region1->getType() == "DateEncoderRegion") << " Expected type for region1 to be \"DateEncoderRegion\" but type is: " << region1->getType();


    EXPECT_THROW(region1->getOutputData("doesnotexist"), std::exception);
    EXPECT_THROW(region1->getInputData("doesnotexist"), std::exception);

    net.initialize();

    time_t t = DateEncoder::mktime(2020, 1, 1); // new years day 2020.
    region1->setParameterInt64("sensedTime", t);
    net.run(1);
	  region1->compute();
	}


  TEST(DateEncoderRegionTest, initialization_with_custom_impl)
  {
    VERBOSE << "Creating network..." << std::endl;
    Network net;

    size_t regionCntBefore = net.getRegions().size();

    // make sure the custom region registration works for CPP.
    // We will just use the same SPRegion class but it could be a subclass or some different custom class.
    // While we are at it, make sure we can initialize the dimensions and parameters from here too.
    // The parameter names and data types must match those of the spec.
    // Explicit parameters:  (Yaml format...but since YAML is a superset of JSON, you can use JSON format as well)
    std::string nodeParams = "{season_width: 5}";

    VERBOSE << "Adding a custom-built DateEncoderRegion region..." << std::endl;
    net.registerRegion("DateEncoderRegionCustom", new RegisteredRegionImplCpp<DateEncoderRegion>());
    std::shared_ptr<Region> region2 = net.addRegion("region2", "DateEncoderRegionCustom", nodeParams);
    size_t regionCntAfter = net.getRegions().size();
    ASSERT_TRUE(regionCntBefore + 1 == regionCntAfter) 
      << "  Expected number of regions to increase by one.  ";
    ASSERT_TRUE(region2->getType() == "DateEncoderRegionCustom") 
      << " Expected type for region2 to be \"DateEncoderRegionCustom\" but type is: " 
      << region2->getType();

    ASSERT_DOUBLE_EQ(region2->getParameterReal32("season_radius"), 91.5f);

    time_t t = DateEncoder::mktime(2020, 04, 17);
    region2->setParameterInt64("sensedTime", t);
    net.run(1);
    region2->compute();
  }




	TEST(DateEncoderRegionTest, testLinking)
	{
    // This is a minimal end-to-end test containing an DateEncoderRegion region.
    // To make sure we can feed data from some other region to our DateEncoderRegion,
    // this test will hook up the VectorFileSensor to our DateEncoderRegion and then
    // connect that to an SPRegion and then on to a VectorFileEffector to capture the results.
    //
    std::string test_input_file = "TestOutputDir/DateEncoderRegionTestInput.csv";


    // make a place to put test data.
    if (!Directory::exists("TestOutputDir")) Directory::create("TestOutputDir", false, true); 
    if (Path::exists(test_input_file)) Path::remove(test_input_file);

    // Create a csv data file to use as input.
    std::ofstream f(test_input_file.c_str());
    f << DateEncoder::mktime(2020, 1, 1, 1, 15, 0) << std::endl;  // Jan 1, 2020 01:15:00
    f << DateEncoder::mktime(2020, 4, 17, 22, 0, 0) << std::endl; // Apr 17, 2020 22:00:00
    f.close();

    VERBOSE << "Setup Network; add 3 regions and 2 links." << std::endl;
	  Network net;

    std::string params = "{timeOfDay_width: 5, holiday_width: 2, holiday_dates: \"[[1,1]]\", verbose: " +
                               std::string((verbose) ? "true" : "false") + "}";
      
    std::shared_ptr<Region> reader = net.addRegion("filereader", "VectorFileSensor", "{activeOutputCount: 1}");
    std::shared_ptr<Region> encoder = net.addRegion("dateEncoder", "DateEncoderRegion", params);
    std::shared_ptr<Region> sp = net.addRegion("sp", "SPRegion", "{columnCount: 200}");


    net.link("filereader", "dateEncoder", "", "", "dataOut", "values");
    net.link("dateEncoder", "sp", "", "", "encoded", "bottomUpIn");

    VERBOSE << "Load Data." << std::endl;
    reader->executeCommand({"loadFile", test_input_file});


    VERBOSE << "Initialize." << std::endl;
    net.initialize();



	  // check actual dimensions
    ASSERT_EQ(encoder->getParameterUInt32("size"), 34u);

    VERBOSE << "Execute" << std::endl;
    net.run(1);

	  VERBOSE << "Checking data after first iteration..." << std::endl;
    Array r1OutputArray = reader->getOutputData("dataOut");
    VERBOSE << "  VectorFileSensor Output" << r1OutputArray << std::endl;
    EXPECT_TRUE(r1OutputArray.getType() == NTA_BasicType_Real32)
            << "actual type is " << BasicType::getName(r1OutputArray.getType());
    VERBOSE << "  " << std::endl;

    Array r2InputArray = encoder->getInputData("values");
    VERBOSE << "  DateEncoderRegion input" << r2InputArray << std::endl;
    ASSERT_TRUE(r1OutputArray.getCount() == r2InputArray.getCount()) 
		<< "Buffer length different. Output from reader is " << r1OutputArray.getCount() << ", input to encoder is " << r2InputArray.getCount();
    
    VERBOSE << "Check the buckets\n";
    Array b = encoder->getOutputData("bucket");
    Array expected_bucket1(std::vector<Real64>({1.0, 0.0}));    // OSx 1, 8
    EXPECT_TRUE(b == expected_bucket1) << "Expected " << expected_bucket1 << " Found " << b;
    
    Array r2OutputArray = encoder->getOutputData("encoded");
    VERBOSE << "  encoder output" << r2OutputArray << std::endl;
    EXPECT_TRUE(r2OutputArray.getType() == NTA_BasicType_SDR) 
      << "actual type is " << BasicType::getName(r2OutputArray.getType());
    SDR expected1({34});
    expected1.setSparse(SDR_sparse_t({2, 3, 6, 7, 8, 9, 10}));  // OSx 2, 3, 16, 17, 18, 19, 20
    EXPECT_TRUE(r2OutputArray.getSDR() == expected1) << "Expected " << expected1 << "  Found: " << r2OutputArray;

    VERBOSE << "Execute one more time.\n";

    net.run(1);

    Array expected_bucket2(std::vector<Real64>({0.0, 20.0}));  // on OSx and linux 0, 0
    EXPECT_TRUE(b == expected_bucket2) << "Expected " << expected_bucket2 << " Found " << b;
    r2OutputArray = encoder->getOutputData("encoded");
    SDR expected2({34});
    expected2.setSparse(SDR_sparse_t({0, 1, 4, 5, 6, 32, 33})); // on OSx and linux 0, 1, 9, 10, 11, 12, 13
    EXPECT_TRUE(r2OutputArray.getSDR() == expected2) << "Expected " << expected2 << "  Found: " << r2OutputArray;


    // cleanup
    Directory::removeTree("TestOutputDir", true);
  }


  TEST(DateEncoderRegionTest, testSerialization) {
    // NOTE: this test does end-to-end serialize and deserialize.
    //
	  // use default parameters for the most part.
	  Network net1;
	  Network net2;
	  Network net3;

	  VERBOSE << "Setup first network and save it" << std::endl;

    std::string params = "{dayOfWeek_width: 5, weekend_width: 5, verbose: "+std::string((verbose)?"true":"false")+"}";

    std::shared_ptr<Region> encoder1 = net1.addRegion("encoder", "DateEncoderRegion", params);
    std::shared_ptr<Region> sp1 = net1.addRegion("sp", "SPRegion", "{columnCount: 200}");
    net1.link("encoder", "sp", "", "", "encoded", "bottomUpIn");
    net1.initialize();

    encoder1->setParameterInt64("sensedTime", DateEncoder::mktime(2020, 1, 1, 1, 15, 0));  // Jan 1, 2020 01:15:00
		net1.run(1);

    std::string filename = "TestOutputDir/spRegionTest.stream";
    Directory::removeTree("TestOutputDir", true);
    VERBOSE << "Writing stream to " << Path::makeAbsolute(filename) << "\n";
	  net1.saveToFile(filename, SerializableFormat::JSON);

    VERBOSE << "Restore from " << Path::makeAbsolute(filename) 
            << " into a second network and compare." << std::endl;
    net2.loadFromFile(filename, SerializableFormat::JSON);

	  std::shared_ptr<Region> encoder2 = net2.getRegion("encoder");
	  std::shared_ptr<Region> sp2 = net2.getRegion("sp");

	  ASSERT_TRUE (encoder2->getType() == "DateEncoderRegion") 
	    << " Restored region does not have the right type.  Expected DateEncoderRegion, found " << encoder2->getType();

    EXPECT_TRUE(net1 == net2);

	  // can we continue with execution?  See if we get any exceptions.
    encoder2->setParameterInt64("sensedTime", DateEncoder::mktime(2020, 4, 17, 22, 0, 0));  // Apr 17, 2020 22:00:00
    net2.run(2);

    // cleanup
    Directory::removeTree("TestOutputDir", true);
	}


} // namespace