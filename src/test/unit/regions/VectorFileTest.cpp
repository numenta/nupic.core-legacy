/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
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
 * Author: David Keeney, April, 2019
 * --------------------------------------------------------------------- */
 
/*---------------------------------------------------------------------
  * This is a test of the VectorFileEffector and VectorFileSensor modules.  
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


#include <htm/engine/NuPIC.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/RegisteredRegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/types/Exception.hpp>
#include <htm/os/Env.hpp>
#include <htm/os/Path.hpp>
#include <htm/os/Timer.hpp>
#include <htm/os/Directory.hpp>
#include <htm/regions/SPRegion.hpp>
#include <htm/utils/LogItem.hpp>


#include <string>
#include <vector>
#include <cmath> // fabs/abs
#include <cstdlib> // exit
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <streambuf>
#include <iterator>
#include <algorithm>



#include "gtest/gtest.h"
#include "RegionTestUtilities.hpp"

#define VERBOSE if(verbose)std::cerr << "[          ] "
static bool verbose = false;  // turn this on to print extra stuff for debugging the test.

// The following string should contain a valid expected Spec - manually verified. 
#define EXPECTED_EFFECTOR_SPEC_COUNT  1  // The number of parameters expected in the VectorFileEffector Spec
#define EXPECTED_SENSOR_SPEC_COUNT  11    // The number of parameters expected in the VectorFileSensor Spec

using namespace htm;
namespace testing 
{

  //  forward declarations;  Find function body below.
  static bool compareFiles(const std::string& p1, const std::string& p2); 
	static void createTestData(size_t dataRows, 
                             size_t dataWidth,
                             const std::string& test_input_file,
                             const std::string& test_output_file);


  // Verify that all parameters are working.
  // Assumes that the default value in the Spec is the same as the default 
  // when creating a region with default constructor.
  TEST(VectorFileTest, testSpecAndParametersEffector)
  {
    Network net;

    // create an Effector region with default parameters
    std::shared_ptr<Region> region1 = net.addRegion("region1", "VectorFileEffector", "");  // use default configuration

		std::set<std::string> excluded;
    checkGetSetAgainstSpec(region1, EXPECTED_EFFECTOR_SPEC_COUNT, excluded, verbose);
    checkInputOutputsAgainstSpec(region1, verbose);

  }
  TEST(VectorFileTest, testSpecAndParametersSensor)
  {
    Network net;

    // create an Sensor region with default parameters
    std::shared_ptr<Region> region1 = net.addRegion("region1", "VectorFileSensor", "");  // use default configuration

    std::set<std::string> excluded = {"scalingMode", "position"};
    checkGetSetAgainstSpec(region1, EXPECTED_SENSOR_SPEC_COUNT, excluded, verbose);
    checkInputOutputsAgainstSpec(region1, verbose);

  }

  
	TEST(VectorFileTest, Seeking)
	{
    std::string test_input_file = "TestOutputDir/TestInput.csv";
    std::string test_output_file = "TestOutputDir/TestOutput.csv";
    size_t dataWidth = 10;
    size_t dataRows = 10;
		createTestData(dataRows, dataWidth, test_input_file, test_output_file);

    Network net;
    std::string params = "{dim: [" + std::to_string(dataWidth) + "]}";
    std::shared_ptr<Region> region1 = net.addRegion("region1", "VectorFileSensor", params);
    EXPECT_EQ(region1->getParameterInt32("position"), -1);

    region1->executeCommand({ "loadFile", test_input_file });
    EXPECT_EQ(region1->getParameterInt32("position"), 9);
    net.run(1);

    EXPECT_EQ(region1->getParameterInt32("position"), 0);
    Array a = region1->getOutputData("dataOut");
    VERBOSE << "1st vector=" << a << std::endl;
    Array expected1(std::vector<Real32>({ 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }));
    EXPECT_TRUE(a == expected1);


    region1->setParameterInt32("position", 5);
    net.run(1);
    EXPECT_EQ(region1->getParameterInt32("position"), 5);
    a = region1->getOutputData("dataOut");
    VERBOSE << "5th vector=" << a << std::endl;
    Array expected5(std::vector<Real32>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f }));
    EXPECT_TRUE(a == expected5);

    // cleanup
    Directory::removeTree("TestOutputDir", true);
  }

	TEST(VectorFileTest, testLinking)
	{
    // This is a minimal end-to-end test containing an Effector and a Sensor region
    // this test will hook up the VectorFileSensor to a VectorFileEffector to capture the results.
    //
    std::string test_input_file = "TestOutputDir/TestInput.csv";
    std::string test_output_file = "TestOutputDir/TestOutput.csv";
    size_t dataWidth = 10;
    size_t dataRows = 10;
		createTestData(dataRows, dataWidth, test_input_file, test_output_file);


    VERBOSE << "Setup Network; add 2 regions and 1 link." << std::endl;
	  Network net;

    // Explicit parameters:  (Yaml format...but since YAML is a superset of JSON, 
    // you can use JSON format as well)

    std::shared_ptr<Region> region1 = net.addRegion("region1", "VectorFileSensor", "{activeOutputCount: 10}");
    std::shared_ptr<Region> region3 = net.addRegion("region3", "VectorFileEffector", "{outputFile: '"+ test_output_file + "'}");


    net.link("region1", "region3", "", "", "dataOut", "dataIn");

    VERBOSE << "Load Data." << std::endl;
    region1->executeCommand({ "loadFile", test_input_file });


    VERBOSE << "Initialize." << std::endl;
    net.initialize();

    VERBOSE << "Execute once." << std::endl;
    //LogItem::setLogLevel(LogLevel_Verbose);
    net.run(1);
    //LogItem::setLogLevel(LogLevel_None);

	  VERBOSE << "Checking data after first iteration..." << std::endl;
    EXPECT_EQ(region1->getParameterInt32("position"), 0);
    VERBOSE << "  VectorFileSensor Output" << std::endl;
    Array r1OutputArray = region1->getOutputData("dataOut");
    EXPECT_EQ(r1OutputArray.getCount(), dataWidth);
    EXPECT_TRUE(r1OutputArray.getType() == NTA_BasicType_Real32)
            << "actual type is " << BasicType::getName(r1OutputArray.getType());

    Array r3InputArray = region3->getInputData("dataIn");
    ASSERT_TRUE(r3InputArray.getType() == NTA_BasicType_Real32)
      << "actual type is " << BasicType::getName(r3InputArray.getType());
    ASSERT_TRUE(r3InputArray.getCount() == dataWidth);

		VERBOSE << "  VectorFileSensor  output: " << r1OutputArray << std::endl;
    VERBOSE << "  VectorFileEffector input: " << r3InputArray << std::endl;
		

	  // execute network several more times and check that it has output.
    VERBOSE << "Execute 9 more times." << std::endl;
    net.run(9);
    EXPECT_EQ(region1->getParameterInt32("position"), 9);


    r1OutputArray = region1->getOutputData("dataOut");
    r3InputArray = region3->getInputData("dataIn");
		VERBOSE << "  VectorFileSensor  output: " << r1OutputArray << std::endl;
    VERBOSE << "  VectorFileEffector input: " << r3InputArray << std::endl;
    EXPECT_TRUE(r3InputArray == r1OutputArray);

    // cleanup
    region3->executeCommand({ "closeFile" });
		
		// Compare files
    ASSERT_TRUE(compareFiles(test_input_file, test_output_file)) << "Files should be the same.";

    VERBOSE << "Execute 1 more time to verify the wrap." << std::endl;
    net.run(1);
    // The current position should be 0 because it wraps at 10.
    EXPECT_EQ(region1->getParameterInt32("position"), 0);
}


TEST(VectorFileTest, testSerialization)
{
    std::string test_input_file = "TestOutputDir/TestInput.csv";
    std::string test_output_file = "TestOutputDir/TestOutput.csv";
    size_t dataWidth = 10;
    size_t dataRows = 10;
		createTestData(dataRows, dataWidth, test_input_file, test_output_file);
		
	  // use default parameters the first time
	  Network net1;
	  Network net3;

	  VERBOSE << "Setup first network and save it" << std::endl;
    std::shared_ptr<Region> n1region1 = net1.addRegion("region1", "VectorFileSensor", "{activeOutputCount: "+std::to_string(dataWidth) +"}");
    std::shared_ptr<Region> n1region3 = net1.addRegion("region3", "VectorFileEffector", "{outputFile: '"+ test_output_file + "'}");
    net1.link("region1", "region3", "", "", "dataOut", "dataIn");

    VERBOSE << "Load Data." << std::endl;
    n1region1->executeCommand({ "loadFile", test_input_file });
    net1.initialize();

		net1.run(1);

    Directory::removeTree("TestOutputDir", true);
	  net1.saveToFile("TestOutputDir/VectorFileTest.stream");

	  VERBOSE << "Restore into a second network and compare." << std::endl;
    net3.loadFromFile("TestOutputDir/VectorFileTest.stream");
	  std::shared_ptr<Region> n3region1 = net3.getRegion("region1");
	  std::shared_ptr<Region> n3region3 = net3.getRegion("region3");

    EXPECT_TRUE(*n1region1.get() == *n3region1.get());
    EXPECT_TRUE(*n1region3.get() == *n3region3.get());

    // cleanup
    Directory::removeTree("TestOutputDir", true);

	}
	
	//////////////////////////////////////////////////////////////////////////////////

	static bool compareFiles(const std::string& p1, const std::string& p2) {
	  std::ifstream f1(p1, std::ifstream::binary|std::ifstream::ate);
	  std::ifstream f2(p2, std::ifstream::binary|std::ifstream::ate);

	  if (f1.fail() || f2.fail()) {
	    return false; //file problem
	  }

	  if (f1.tellg() != f2.tellg()) {
	    return false; //size mismatch
	  }

	  //seek back to beginning and use std::equal to compare contents
	  f1.seekg(0, std::ifstream::beg);
	  f2.seekg(0, std::ifstream::beg);
	  return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
	                    std::istreambuf_iterator<char>(),
	                    std::istreambuf_iterator<char>(f2.rdbuf()));
	}
	
	static void createTestData(size_t dataRows, size_t dataWidth,
                             const std::string& test_input_file,
                             const std::string& test_output_file) {
	    // make a place to put test data.
    if (!Directory::exists("TestOutputDir")) Directory::create("TestOutputDir", false, true); 
    if (Path::exists(test_input_file)) Path::remove(test_input_file);
    if (Path::exists(test_output_file)) Path::remove(test_output_file);

    // Create a csv file to use as input.
    // The SDR data we will feed it will be a matrix with 1's on the diagonal
    // and we will feed it one row at a time, for 'dataRows' rows.
    std::ofstream  f(test_input_file.c_str());
    for (size_t i = 0; i < dataRows; i++) {
      for (size_t j = 0; j < dataWidth; j++) {
        if (j > 0) f << ",";
        if ((j % dataRows) == i) f << "1";
        else f << "0";
      }
      f << std::endl;
    }
    f.close();
  }


} // namespace

