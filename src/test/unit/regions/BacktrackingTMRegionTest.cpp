/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
 

 * Author: David Keeney, April, 2018
 * ---------------------------------------------------------------------
 */

/*---------------------------------------------------------------------
 * This is a test of the TMRegion module.  It does not check the
 *BacktrackingTMCpp itself but rather just the plug-in mechanisom to call the
 *BacktrackingTMCpp.
 *
 * For those not familiar with GTest:
 *     ASSERT_TRUE(value)   -- Fatal assertion that the value is true.  Test
 *terminates if false. ASSERT_FALSE(value)   -- Fatal assertion that the value
 *is false. Test terminates if true. ASSERT_STREQ(str1, str2)   -- Fatal
 *assertion that the strings are equal. Test terminates if false.
 *
 *     EXPECT_TRUE(value)   -- Nonfatal assertion that the value is true.  Test
 *continues if false. EXPECT_FALSE(value)   -- Nonfatal assertion that the value
 *is false. Test continues if true. EXPECT_STREQ(str1, str2)   -- Nonfatal
 *assertion that the strings are equal. Test continues if false.
 *
 *     EXPECT_THROW(statement, exception_type) -- nonfatal exception, cought and
 *continues.
 *---------------------------------------------------------------------
 */

#include <nupic/engine/Input.hpp>
#include <nupic/engine/Link.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/engine/YAMLUtils.hpp>
#include <nupic/math/Math.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp> // memory leak detection
#include <nupic/os/Path.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/regions/TMRegion.hpp>
#include <nupic/types/Exception.hpp>

#include <cmath>   // fabs/abs
#include <cstdlib> // exit
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include "regionTestUtilities.hpp"
#include "yaml-cpp/yaml.h"
#include "gtest/gtest.h"

#define VERBOSE                                                                \
  if (verbose)                                                                 \
  std::cerr << "[          ] "
static bool verbose = true; // turn this on to print extra stuff for debugging the test.

// The following string should contain a valid expected Spec - manually
// verified.
#define EXPECTED_SPEC_COUNT 38 // The number of parameters expected in the TMRegion Spec

using namespace nupic;
namespace testing {

// Verify that all parameters are working.
// Assumes that the default value in the Spec is the same as the default when
// creating a region with default constructor. 
TEST(TMRegionTest, testSpecAndParameters) {
  Network net;

  // create a TM region with default parameters
  Region_Ptr_t region1 = net.addRegion("region1", "TMRegion", ""); 
  checkGetSetAgainstSpec(region1, EXPECTED_SPEC_COUNT, verbose);
  checkInputOutputsAgainstSpec(region1, verbose);
}

TEST(TMRegionTest, checkTMRegionImpl) {
  Network net;

  size_t regionCntBefore = net.getRegions().getCount();

  VERBOSE << "Adding a built-in TMRegion region..." << std::endl;
  Region_Ptr_t region1 = net.addRegion("region1", "TMRegion", "");
  size_t regionCntAfter = net.getRegions().getCount();
  ASSERT_TRUE(regionCntBefore + 1 == regionCntAfter)
      << " Expected number of regions to increase by one.  ";
  ASSERT_TRUE(region1->getType() == "TMRegion")
      << " Expected type for region1 to be \"TMRegion\" but type is: "
      << region1->getType();

  EXPECT_THROW(region1->getOutputData("doesnotexist"), std::exception);
  EXPECT_THROW(region1->getInputData("doesnotexist"), std::exception);

  // run and compute() should fail because network has not been initialized
  EXPECT_THROW(net.run(1), std::exception);
  EXPECT_THROW(region1->compute(), std::exception);
}

TEST(TMRegionTest, initialization_with_custom_impl) {
  VERBOSE << "Creating network..." << std::endl;
  Network net;

  size_t regionCntBefore = net.getRegions().getCount();

  // make sure the custom region registration works for CPP.
  // We will just use the same TMRegion class but it could be a subclass or some
  // different custom class. While we are at it, let's make sure we can initialize the
  // parameters from here too. The parameter names and data types
  // must match those of the spec. Explicit parameters are Yaml format...but
  // since YAML is a superset of JSON, you can use JSON format as well.
  // Here we set a unique value for every parameter we can set (per the spec).
  std::string nodeParams =  
      "{numberOfCols: 100, cellsPerColumn: 20, initialPerm: 0.12, "
      "connectedPerm: 0.6, minThreshold: 9, newSynapseCount: 16, "
      "permanenceInc: 0.2, permanenceDec: 0.2, permanenceMax: 2.0, "
      "globalDecay: 0.2, activationThreshold: 13, doPooling: true, "
      "segUpdateValidDuration: 6, burnIn: 1, collectStats: true, "
      "seed: 66, verbosity: 3, checkSynapseConsistency: true, "
      "pamLength: 2, maxInfBacktrack: 9, maxLrnBacktrack: 6, "
      "maxAge: 99999, maxSeqLength: 32, maxSegmentsPerCell: 3, "
      "maxSynapsesPerSegment: 20, outputType: activeState1CellPerCol, "
      "learningMode: false, inferenceMode: true, anomalyMode: true, "
      "topDownMode: true, storeDenseOutput: true, computePredictedActiveCellIndices: true, "
      "orColumnOutputs: true, cellsSavePath: xxx, logPathOutput: yyy, temporalImp: zzz}";

  VERBOSE << "Adding a custom-built TMRegion region..." << std::endl;
  net.registerCPPRegion("TMRegionCustom", new RegisteredRegionImplCpp<TMRegion>());
  Region_Ptr_t region2 = net.addRegion("region2", "TMRegionCustom", nodeParams);
  size_t regionCntAfter = net.getRegions().getCount();
  ASSERT_TRUE(regionCntBefore + 1 == regionCntAfter)
      << "  Expected number of regions to increase by one.  ";
  ASSERT_TRUE(region2->getType() == "TMRegionCustom")
      << " Expected type for region2 to be \"TMRegionCustom\" but type is: "
      << region2->getType();

  // Check that all of the node parameters have been correctly parsed and available.
  EXPECT_EQ(region2->getParameterUInt32("numberOfCols"), 100);
  EXPECT_EQ(region2->getParameterUInt32("cellsPerColumn"), 20);
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("initialPerm"), 0.12f));
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("connectedPerm"), 0.6f)); 
  EXPECT_EQ(region2->getParameterUInt32("minThreshold"), 9);
  EXPECT_EQ(region2->getParameterUInt32("newSynapseCount"), 16);
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("permanenceInc"), 0.2f)); 
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("permanenceDec"), 0.2f)); 
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("permanenceMax"), 2.0f));
  EXPECT_TRUE(nearlyEqual<Real32>(region2->getParameterReal32("globalDecay"), 0.2f));
  EXPECT_EQ(region2->getParameterUInt32("activationThreshold"), 13); 
  EXPECT_EQ(region2->getParameterBool("doPooling"), true);
  EXPECT_EQ(region2->getParameterUInt32("segUpdateValidDuration"), 6); 
  EXPECT_EQ(region2->getParameterUInt32("burnIn"), 1); 
  EXPECT_EQ(region2->getParameterBool("collectStats"), true);
  EXPECT_EQ(region2->getParameterInt32("seed"), 66); 
  EXPECT_EQ(region2->getParameterUInt32("verbosity"), 3); 
  EXPECT_EQ(region2->getParameterBool("checkSynapseConsistency"), true);
  EXPECT_EQ(region2->getParameterUInt32("pamLength"), 2); 
  EXPECT_EQ(region2->getParameterUInt32("maxInfBacktrack"), 9); 
  EXPECT_EQ(region2->getParameterUInt32("maxLrnBacktrack"), 6); 
  EXPECT_EQ(region2->getParameterUInt32("maxAge"), 99999); 
  EXPECT_EQ(region2->getParameterUInt32("maxSeqLength"), 32); 
  EXPECT_EQ(region2->getParameterInt32("maxSegmentsPerCell"), 3);
  EXPECT_EQ(region2->getParameterInt32("maxSynapsesPerSegment"), 20);
  EXPECT_STREQ(region2->getParameterString("outputType").c_str(), "activeState1CellPerCol");
  EXPECT_EQ(region2->getParameterBool("learningMode"), false);
  EXPECT_EQ(region2->getParameterBool("inferenceMode"), true);
  EXPECT_EQ(region2->getParameterBool("anomalyMode"), true);
  EXPECT_EQ(region2->getParameterBool("topDownMode"), true);
  EXPECT_EQ(region2->getParameterBool("storeDenseOutput"), true);
  EXPECT_EQ(region2->getParameterBool("computePredictedActiveCellIndices"), true);
  EXPECT_EQ(region2->getParameterBool("orColumnOutputs"), true);
  EXPECT_STREQ(region2->getParameterString("cellsSavePath").c_str(), "xxx");
  EXPECT_STREQ(region2->getParameterString("logPathOutput").c_str(), "yyy");
  EXPECT_STREQ(region2->getParameterString("temporalImp").c_str(), "zzz");

  // compute() should fail because network has not been initialized
  EXPECT_THROW(net.run(1), std::exception);
  EXPECT_THROW(region2->compute(), std::exception);

  EXPECT_THROW(net.initialize(), std::exception)
      << "Exception should say region2 has unspecified dimensions. ";
}

TEST(TMRegionTest, testLinking) {
  // This is a minimal end-to-end test containing an TMRegion region.
  // To make sure we can feed data from some other region to our TMRegion
  // this test will hook up the VectorFileSensor to our TMRegion and then
  // connect our TMRegion to a VectorFileEffector to capture the results.
  //
  std::string test_input_file = "TestOutputDir/TMRegionTestInput.csv";
  std::string test_output_file = "TestOutputDir/TMRegionTestOutput.csv";

  // make a place to put test data.
  if (!Directory::exists("TestOutputDir"))
    Directory::create("TestOutputDir", false, true);
  if (Path::exists(test_input_file))
    Path::remove(test_input_file);
  if (Path::exists(test_output_file))
    Path::remove(test_output_file);

  // Create a csv file to use as input.
  // The SDR data we will feed it will be a matrix with 1's on the diagonal
  // and we will feed it one row at a time, for 10 rows.
  size_t dataWidth = 10;
  size_t dataRows = 10;
  std::ofstream f(test_input_file.c_str());
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < dataWidth; j++) {
      if ((j % dataRows) == i)
        f << "1.0,";
      else
        f << "0.0,";
    }
    f << std::endl;
  }
  f.close();

  VERBOSE << "Setup Network; add 3 regions and 2 links." << std::endl;
  Network net;

  // Explicit parameters:  (Yaml format...but since YAML is a superset of JSON,
  // you can use JSON format as well)

  Region_Ptr_t region1 =
      net.addRegion("region1", "VectorFileSensor",
                    "{activeOutputCount: " + std::to_string(dataWidth) + "}");
  Region_Ptr_t region2 =
      net.addRegion("region2", "TMRegion", "{numberOfCols: 10, }");
  Region_Ptr_t region3 =
      net.addRegion("region3", "VectorFileEffector",
                    "{outputFile: '" + test_output_file + "'}");

  net.link("region1", "region2", "UniformLink", "", "dataOut", "bottomUpIn");
  net.link("region2", "region3", "UniformLink", "", "bottomUpOut", "dataIn");

  VERBOSE << "Load Data." << std::endl;
  region1->executeCommand({"loadFile", test_input_file});

  VERBOSE << "Initialize." << std::endl;
  net.initialize();

  // check actual dimensions
  ASSERT_EQ(region2->getParameterUInt32("numberOfCols"), 10);
  ASSERT_EQ(region2->getParameterUInt32("inputWidth"), (UInt32)dataWidth);

  VERBOSE << "Execute once." << std::endl;
  net.run(1);

  VERBOSE << "Checking data after first iteration..." << std::endl;
  VERBOSE << "  VectorFileSensor Output" << std::endl;
  ArrayRef r1OutputArray = region1->getOutputData("dataOut");
  EXPECT_EQ(r1OutputArray.getCount(), dataWidth);
  EXPECT_TRUE(r1OutputArray.getType() == NTA_BasicType_Real32);
  const Real32 *buffer1 = (const Real32 *)r1OutputArray.getBuffer();

  VERBOSE << "  TMRegion input" << std::endl;
  ArrayRef r2InputArray = region2->getInputData("bottomUpIn");
  ASSERT_TRUE(r1OutputArray.getCount() == r2InputArray.getCount())
      << "Buffer length different. Output from VectorFileSensor is "
      << r1OutputArray.getCount() << ", input to TPRegion is "
      << r2InputArray.getCount();
  EXPECT_TRUE(r2InputArray.getType() == NTA_BasicType_Real32);
  const Real32 *buffer2 = (const Real32 *)r2InputArray.getBuffer();
  for (size_t i = 0; i < r2InputArray.getCount(); i++) {
    // VERBOSE << "  [" << i << "]=    " << buffer2[i] << "" << std::endl;
    ASSERT_TRUE(buffer2[i] == buffer1[i])
        << " Buffer content different. Element " << i
        << " of Output from encoder is " << buffer1[i]
        << ", input to SPRegion is " << buffer2[i];
    ASSERT_TRUE(buffer2[i] == 1.0f || buffer2[i] == 0.0f)
        << " Value[" << i << "] is not a 0 or 1." << std::endl;
  }

  // execute TMRegion several more times and check that it has output.
  VERBOSE << "Execute 9 times." << std::endl;
  net.run(9);

  VERBOSE << "Checking Output Data." << std::endl;
  VERBOSE << "  TMRegion output" << std::endl;
  UInt32 columnCount = region2->getParameterUInt32("numberOfCols");
  UInt32 cellsPerColumn = region2->getParameterUInt32("cellsPerColumn");
  UInt32 nCells = columnCount * cellsPerColumn;
  ArrayRef r2OutputArray = region2->getOutputData("bottomUpOut");
  ASSERT_TRUE(r2OutputArray.getCount() == nCells)
      << "Buffer length different. Output from TMRegion is "
      << r2OutputArray.getCount() << ", should be " << nCells;
  const Real32 *buffer3 = (const Real32 *)r2OutputArray.getBuffer();
  for (size_t i = 0; i < r2OutputArray.getCount(); i++) {
    // VERBOSE << "  [" << i << "]=    " << buffer3[i] << "" << std::endl;
    ASSERT_TRUE(buffer3[i] == 0.0f || buffer3[i] == 1.0f)
        << " Element " << i
        << " of Output from SPRegion is not 0.0 or 1.0; it is " << buffer3[i];
  }

  VERBOSE << "  VectorFileEffector input" << std::endl;
  ArrayRef r3InputArray = region3->getInputData("dataIn");
  ASSERT_TRUE(r3InputArray.getCount() == nCells);
  const Real32 *buffer4 = (const Real32 *)r3InputArray.getBuffer();
  for (size_t i = 0; i < r3InputArray.getCount(); i++) {
    // VERBOSE << "  [" << i << "]=    " << buffer4[i] << "" << std::endl;
    ASSERT_TRUE(buffer3[i] == buffer4[i])
        << " Buffer content different. Element[" << i
        << "] from SPRegion out is " << buffer3[i]
        << ", input to VectorFileEffector is " << buffer4[i];
  }

  // cleanup
  region3->executeCommand({"closeFile"});
}

TEST(TMRegionTest, testSerialization) {
  // use default parameters the first time
  Network *net1 = new Network();
  Network *net2 = nullptr;
  Network *net3 = nullptr;

  try {

    VERBOSE << "Setup first network and save it" << std::endl;
    Region_Ptr_t n1region1 = net1->addRegion( "region1", "ScalarSensor", 
                                             "{n: 100,w: 10,minValue: 0,maxValue: 10}");
    n1region1->setParameterReal64("sensedValue", 5.0);

    Region_Ptr_t n1region2 =  net1->addRegion("region2", "TMRegion", "{numberOfCols: 48}");

    net1->link("region1", "region2", "", "", "encoded", "bottomUpIn");
    net1->initialize();

    n1region1->prepareInputs();
    n1region1->compute();

    n1region2->prepareInputs();
    n1region2->compute();

    // take a snapshot of everything in TMRegion at this point
    // save to a bundle.
    std::map<std::string, std::string> parameterMap;
    EXPECT_TRUE(captureParameters(n1region2, parameterMap))
        << "Capturing parameters before save.";

    Directory::removeTree("TestOutputDir", true);
    net1->saveToFile("TestOutputDir/tmRegionTest.stream");

    VERBOSE << "Restore from bundle into a second network and compare." << std::endl;
    net2 = new Network();
    net2->loadFromFile("TestOutputDir/tmRegionTest.stream");

    Region_Ptr_t n2region2 = net2->getRegions().getByName("region2");
    ASSERT_TRUE(n2region2->getType() == "TMRegion")
        << " Restored TMRegion region does not have the right type.  Expected "
           "TMRegion, found "
        << n2region2->getType();

    EXPECT_TRUE(compareParameters(n2region2, parameterMap))
        << "Conflict when comparing TMRegion parameters after restore with "
           "before save.";

    // can we continue with execution?  See if we get any exceptions.
    n1region1->setParameterReal64("sensedValue", 0.12);
    n1region1->prepareInputs();
    n1region1->compute();

    n2region2->prepareInputs();
    n2region2->compute();

    // Change some parameters and see if they are retained after a restore.
    n2region2->setParameterBool("collectStats", true);
    n2region2->setParameterUInt32("pamLength", 3);
    n2region2->compute();

    parameterMap.clear();
    EXPECT_TRUE(captureParameters(n2region2, parameterMap))
        << "Capturing parameters before second save.";
    // serialize using a stream to a single file
    net2->saveToFile("TestOutputDir/tmRegionTest.stream");

    VERBOSE << "Restore into a third network and compare changed parameters."
            << std::endl;
    net3 = new Network();
    net3->loadFromFile("TestOutputDir/tmRegionTest.stream");
    Region_Ptr_t n3region2 = net3->getRegions().getByName("region2");
    EXPECT_TRUE(n3region2->getType() == "TMRegion")
        << "Failure: Restored region does not have the right type. "
           " Expected \"TMRegion\", found \""
        << n3region2->getType() << "\".";

    EXPECT_TRUE(compareParameters(n3region2, parameterMap))
        << "Comparing parameters after second restore with before save.";
  } catch (nupic::Exception &ex) {
    FAIL() << "Failure: Exception: " << ex.getFilename() << "("
           << ex.getLineNumber() << ") " << ex.getMessage() << "" << std::endl;
  } catch (std::exception &e) {
    FAIL() << "Failure: Exception: " << e.what() << "" << std::endl;
  }

  // cleanup
  if (net1 != nullptr) {
    delete net1;
  }
  if (net2 != nullptr) {
    delete net2;
  }
  if (net3 != nullptr) {
    delete net3;
  }
  Directory::removeTree("TestOutputDir", true);
}

TEST(TMRegionTest, checkTMRegionIO) {
  Network *net1 = new Network();
  try {
    Region_Ptr_t n1region1 = net1->addRegion("region1", "TMRegion", 
      "{numberOfCols: 10, cellsPerColumn: 4, learningMode: true, collectStats: true}");
    n1region1->getOutput("bottomUpOut")->getData().allocateBuffer(40);
    n1region1->getOutput("bottomUpOut")->getData().zeroBuffer();
    n1region1->getOutput("topDownOut")->getData().allocateBuffer(10);
    n1region1->getOutput("topDownOut")->getData().zeroBuffer();
    n1region1->getOutput("activeCells")->getData().allocateBuffer(40);
    n1region1->getOutput("activeCells")->getData().zeroBuffer();
    n1region1->getOutput("predictedActiveCells")->getData().allocateBuffer(40);
    n1region1->getOutput("predictedActiveCells")->getData().zeroBuffer();
    n1region1->getOutput("anomalyScore")->getData().allocateBuffer(1);
    n1region1->getOutput("anomalyScore")->getData().zeroBuffer();
    n1region1->getOutput("lrnActiveStateT")->getData().allocateBuffer(40);
    n1region1->getOutput("lrnActiveStateT")->getData().zeroBuffer();

    // manually allocate the input buffers (normally done by link during init)
    n1region1->getInput("bottomUpIn")->initialize();
    n1region1->getInput("bottomUpIn")->getData().allocateBuffer(10);
    n1region1->getInput("resetIn")->initialize();
    n1region1->getInput("resetIn")->getData().allocateBuffer(1);
    n1region1->getInput("sequenceIdIn")->initialize();
    n1region1->getInput("sequenceIdIn")->getData().allocateBuffer(1);

    // initialize inputs
    // Note: inputs are not saved in serialization...only outputs
    //       so serialization of a network with manually initialized buffers
    //       will not work.
    Real32 a[10] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    Real32 b[10] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    ((Real32 *)n1region1->getInput("resetIn")->getData().getBuffer())[0] = 1.0f;
    ((UInt64 *)n1region1->getInput("sequenceIdIn")->getData().getBuffer())[0] = 1;

    n1region1->initialize();
    n1region1->compute();

    for (Size i = 0; i < 10; i++) {
      // change input data and execute
      memcpy(n1region1->getInput("bottomUpIn")->getData().getBuffer(), a, 10 * sizeof(Real32));
      n1region1->compute();

      // change input data and execute again.
      memcpy(n1region1->getInput("bottomUpIn")->getData().getBuffer(), b, 10 * sizeof(Real32));
      n1region1->compute();
    }

    // The outputs could vary depending on the algorithm.
    // The best we can do for this test is to just be sure there is
    // an output.
    //cout << "bottomUpOut " << n1region1->getOutput("bottomUpOut")->getData() << std::endl;
    EXPECT_TRUE(n1region1->getOutput("bottomUpOut")->getData().getCount() > 0);
    //cout << "topDownOut " << n1region1->getOutput("topDownOut")->getData() << std::endl;
    EXPECT_TRUE(n1region1->getOutput("topDownOut")->getData().nonZero().getCount() == 0);
    //cout << "activeCells " << n1region1->getOutput("activeCells")->getData() << std::endl;
    EXPECT_TRUE(n1region1->getOutput("activeCells")->getData().nonZero().getCount() == 0);
    //cout << "predictedActiveCells " << n1region1->getOutput("predictedActiveCells")->getData() << std::endl;
    EXPECT_TRUE(n1region1->getOutput("predictedActiveCells")->getData().nonZero().getCount() == 0);
    //cout << "anomalyScore " << n1region1->getOutput("anomalyScore")->getData() << std::endl;
    EXPECT_TRUE(((Real32*)n1region1->getOutput("anomalyScore")->getData().getBuffer())[0] == 0);
    //cout << "lrnActiveStateT " << n1region1->getOutput("lrnActiveStateT")->getData() << std::endl;
    EXPECT_TRUE(n1region1->getOutput("lrnActiveStateT")->getData().nonZero().getCount() ==  0);
  } catch (nupic::Exception &ex) {
    FAIL() << "Failure: Exception: " << ex.getFilename() << "("
           << ex.getLineNumber() << ") " << ex.getMessage() << "" << std::endl;
  } catch (std::exception &e) {
    FAIL() << "Failure: Exception: " << e.what() << "" << std::endl;
  }

  delete net1;
}

} // namespace testing
