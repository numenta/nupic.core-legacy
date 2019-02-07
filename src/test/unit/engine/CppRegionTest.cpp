/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

#include "gtest/gtest.h"

#include <nupic/engine/Input.hpp>
#include <nupic/engine/Link.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/types/Exception.hpp>

#include <cmath>   // fabs/abs
#include <cstdlib> // exit
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>  // for max()


namespace testing {

using namespace nupic;
using std::exception;

bool verbose = false;

void helperCppInputOutputAccess(Region *level1) {
  // --- input/output access for level 1 (C++ TestNode) ---

//  SHOULDFAIL(level1->getOutputData("doesnotexist"));

  // getting access via zero-copy
  std::cout << "Getting output for zero-copy access" << std::endl;
  const Array& output = level1->getOutputData("bottomUpOut");
  std::cout << "Element count in bottomUpOut is " << output.getCount() << ""
            << std::endl;
  Real64 *data_actual = (Real64 *)output.getBuffer();
  // set the actual output
  data_actual[12] = 54321;
}


TEST(CppRegionTest, testCppLinkingFanIn) {
  Network net;

  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "");

  net.link("region1", "region2", "TestFanIn2", ""); //the only change testCppLinking* is here

  if (verbose) std::cout << "Initialize should fail..." << std::endl;
  EXPECT_THROW(net.initialize(), exception);

  if (verbose) std::cout << "Setting region1 dims" << std::endl;
  Dimensions r1dims;
  r1dims.push_back(6);
  r1dims.push_back(4);
  region1->setDimensions(r1dims);

  if (verbose) std::cout << "Initialize should now succeed" << std::endl;
  net.initialize();

  const Dimensions &r2dims = region2->getDimensions();
  EXPECT_EQ(r2dims.size(), 2u) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[0], 3u) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[1], 2u) << " actual dims: " << r2dims.toString();

  EXPECT_THROW(region2->setDimensions(r1dims), exception);

  const Array r1OutputArray = region1->getOutputData("bottomUpOut");

  if (verbose) std::cout << "region1 Dims: " << region1->getDimensions() << std::endl;
  if (verbose) std::cout << "region2 Dims: " << region2->getDimensions() << std::endl;

  region1->compute();
  if (verbose) std::cout << "Checking region1 output after first iteration..." << std::endl;
  if (verbose) std::cout << r1OutputArray << std::endl;
  Real64 *buffer = (Real64 *)r1OutputArray.getBuffer();
  for (size_t i = 0; i < r1OutputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  region2->prepareInputs();
  const Array r2InputArray = region2->getInputData("bottomUpIn");
  if (verbose) std::cout << "Region 2 input after first iteration:" << std::endl;
  Real64 *buffer2 = (Real64 *)r2InputArray.getBuffer();
  if (verbose) std::cout << r2InputArray << std::endl;
  EXPECT_TRUE(buffer == buffer2);

  for (size_t i = 0; i < r2InputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  if (verbose) std::cout << "Region 2 input by node" << std::endl;
  std::vector<Real64> r2NodeInput;

  for (size_t node = 0; node < 6; node++) {
    region2->getInput("bottomUpIn")->getInputForNode(node, r2NodeInput);
    if (verbose) {
      std::cout << "Node " << node << ": ";
      for (auto &elem : r2NodeInput) {
        std::cout << elem << " ";
      }
      std::cout << "" << std::endl;
    }
    // 4 nodes in r1 fan in to 1 node in r2
    int row = (int)node / 3;
    int col = (int)node - (row * 3);
    EXPECT_EQ(r2NodeInput.size(), 8u);
    EXPECT_EQ(r2NodeInput[0], 0);
    EXPECT_EQ(r2NodeInput[2], 0);
    EXPECT_EQ(r2NodeInput[4], 0);
    EXPECT_EQ(r2NodeInput[6], 0);
    // these values are specific to the fanin2 link policy
    EXPECT_EQ(r2NodeInput[1], row * 12 + col * 2)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[1];
    EXPECT_EQ(r2NodeInput[3], row * 12 + col * 2 + 1)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[3];
    EXPECT_EQ(r2NodeInput[5], row * 12 + 6 + col * 2)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[5];
    EXPECT_EQ(r2NodeInput[7], row * 12 + 6 + col * 2 + 1)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[7];
  }
}


TEST(CppRegionTest, testCppLinkingUniformLink) {
  Network net;

  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "");

  net.link("region1", "region2", "UniformLink", "{mapping: in, rfSize: [2]}"); //the only change testCppLinking* is here


  if (verbose) std::cout << "Initialize should fail..." << std::endl;
  EXPECT_THROW(net.initialize(), exception);

  if (verbose) std::cout << "Setting region1 dims" << std::endl;
  Dimensions r1dims;
  r1dims.push_back(6);
  r1dims.push_back(4);
  region1->setDimensions(r1dims);

  if (verbose) std::cout << "Initialize should now succeed" << std::endl;
  net.initialize();

  const Dimensions &r2dims = region2->getDimensions();
  EXPECT_EQ(r2dims.size(), 2u) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[0], 3u) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[1], 2u) << " actual dims: " << r2dims.toString();

  EXPECT_THROW(region2->setDimensions(r1dims), exception);

  const Array r1OutputArray = region1->getOutputData("bottomUpOut");

  region1->compute();
  if (verbose) std::cout << "Checking region1 output after first iteration..." << std::endl;
  Real64 *buffer = (Real64 *)r1OutputArray.getBuffer();
  for (size_t i = 0; i < r1OutputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  region2->prepareInputs();
  const Array r2InputArray = region2->getInputData("bottomUpIn");
  if (verbose) std::cout << "Region 2 input after first iteration:" << std::endl;
  Real64 *buffer2 = (Real64 *)r2InputArray.getBuffer();
  // only one link to this input so buffer is passed without copy.
  EXPECT_TRUE(buffer == buffer2);  // 
  for (size_t i = 0; i < r2InputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer2[i], 0);
    else
      ASSERT_EQ(buffer2[i], (i - 1) / 2);
  }

  if (verbose) std::cout << "Region 2 input by node" << std::endl;
  std::vector<Real64> r2NodeInput;

  for (size_t node = 0; node < 6; node++) {
    region2->getInput("bottomUpIn")->getInputForNode(node, r2NodeInput);
    if (verbose) {
      std::cout << "Node " << node << ": ";
      for (auto &elem : r2NodeInput) {
        std::cout << elem << " ";
      }
      std::cout << "" << std::endl;
    }
    // 4 nodes in r1 fan in to 1 node in r2
    int row = (int)node / 3;
    int col = (int)node - (row * 3);
    EXPECT_EQ(r2NodeInput.size(), 8u);
    EXPECT_EQ(r2NodeInput[0], 0);
    EXPECT_EQ(r2NodeInput[2], 0);
    EXPECT_EQ(r2NodeInput[4], 0);
    EXPECT_EQ(r2NodeInput[6], 0);
    // these values are specific to the link policy
    EXPECT_EQ(r2NodeInput[1], row * 12 + col * 2)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[1];
    EXPECT_EQ(r2NodeInput[3], row * 12 + col * 2 + 1)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[3];
    EXPECT_EQ(r2NodeInput[5], row * 12 + 6 + col * 2)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[5];
    EXPECT_EQ(r2NodeInput[7], row * 12 + 6 + col * 2 + 1)
        << "row: " << row << " col: " << col << " val: " << r2NodeInput[7];
  }
}



TEST(CppRegionTest, testYAML) {
  const char *params = "{int32Param: 1234, real64Param: 23.1}";
  //  badparams contains a non-existent parameter
  const char *badparams = "{int32Param: 1234, real64Param: 23.1, badParam: 4}";

  Network net;
  std::shared_ptr<Region> level1;
  EXPECT_THROW(net.addRegion("level1", "TestNode", badparams), exception);

  EXPECT_NO_THROW({level1 = net.addRegion("level1", "TestNode", params);});
  Dimensions d;
  d.push_back(1);
  level1->setDimensions(d);
  net.initialize();

  // check default values
  Real32 r32val = level1->getParameterReal32("real32Param");
  EXPECT_NEAR(r32val, 32.1f, 0.00001) << "r32val = " << r32val << " diff = " << (r32val - 32.1);

  Int64 i64val = level1->getParameterInt64("int64Param");
  EXPECT_EQ(i64val,  64) << "i64val = " << i64val;

  // check values set in region constructor
  Int32 ival = level1->getParameterInt32("int32Param");
  EXPECT_EQ(ival, 1234) << "ival = " << ival;
  Real64 rval = level1->getParameterReal64("real64Param");
  EXPECT_NEAR(rval, 23.1, 0.00000000001) << "rval = " << rval;
  // TODO: if we get the real64 param with getParameterInt32
  // it works -- should we flag an error?

  if (verbose) std::cout
      << "Got the correct values for all parameters set at region creation"
      << std::endl;
}



void helperRealmain(Network& n) {
  // verbose == true turns on extra output that is useful for
  // debugging the test (e.g. when the TestNode compute()
  // algorithm changes)

  size_t count = n.getRegions().getCount();
  EXPECT_TRUE(count == 0);
  std::shared_ptr<Region> level1 = n.addRegion("level1", "TestNode", "");
  count = n.getRegions().getCount();
  EXPECT_TRUE(count == 1);
  std::string region_type = level1->getType();
  EXPECT_STREQ(region_type.c_str(), "TestNode");
  std::string ns = level1->getSpec()->toString();
  EXPECT_GT(ns.length(), 20u); // make sure we got something.

  Int64 val;
  Real64 rval;
  std::string int64Param("int64Param");
  std::string real64Param("real64Param");

  val = level1->getParameterInt64(int64Param);
  EXPECT_EQ(val, 64) << "Default value for int64Param should be 64";
  rval = level1->getParameterReal64(real64Param);
  EXPECT_DOUBLE_EQ(rval, 64.1) << "Default value for real64Param should be 64.1";

  val = 20;
  level1->setParameterInt64(int64Param, val);
  val = level1->getParameterInt64(int64Param);
  EXPECT_EQ(val, 20) << "Value for level1.int64Param not set to 20.";

  rval = 30.1;
  level1->setParameterReal64(real64Param, rval);
  rval = level1->getParameterReal64(real64Param);
  EXPECT_DOUBLE_EQ(rval, 30.1) << " after setting to 30.1";

  // --- test getParameterInt64Array ---
  // Array a is not allocated by us. Will be allocated inside getParameter
  Array a(NTA_BasicType_Int64);
  level1->getParameterArray("int64ArrayParam", a);
  EXPECT_EQ(a.getCount(), 4u) << "size set in initializer of TestNode is 4";
  Int64 expected1[4] = { 0, 64, 128, 192 };
  Int64 *buff = (Int64 *)a.getBuffer();
  for (size_t i = 0; i < std::max<size_t>(a.getCount(), 4u); ++i)
    EXPECT_EQ(buff[i], expected1[i]) << "Invalid value for index " << i;

  // --- test setParameterInt64Array ---
  std::vector<Int64> v(5);
  for (size_t i = 0; i < 5; ++i)
    v[i] = i + 10;
  Array newa(NTA_BasicType_Int64, &v[0], v.size());
  level1->setParameterArray("int64ArrayParam", newa);

  // get the value of intArrayParam after the setParameter call.

  // The array a owns its buffer, so we can call releaseBuffer if we
  // want, but the buffer should be reused if we just pass it again.
  // a.releaseBuffer();
  level1->getParameterArray("int64ArrayParam", a);
  EXPECT_EQ(a.getCount(), 5u) << "set size of TestNode::int64ArrayParam is 5";
  buff = (Int64 *)a.getBuffer();
  for (size_t i = 0; i < std::max<size_t>(a.getCount(), 5u); ++i)
    EXPECT_EQ(buff[i], v[i]) << "Invalid value for index " << i;

}


TEST(CppRegionTest, realmain) {
  Network n;
  helperRealmain(n);

  // should fail because network has not been initialized
  EXPECT_THROW(n.run(1), exception);
  // should fail because network can't be initialized
  EXPECT_THROW(n.initialize(), exception);

  if (verbose) std::cout << "Setting dimensions of level1..." << std::endl;
  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  std::shared_ptr<Region> level1 = n.getRegion("level1");
  level1->setDimensions(d);

  if (verbose) std::cout << "Initializing again..." << std::endl;
  EXPECT_NO_THROW(n.initialize());

  level1->compute();

  EXPECT_NO_THROW(helperCppInputOutputAccess(level1.get()));
  EXPECT_THROW(level1->getOutputData("doesnotexist"), exception);

}


} //ns
