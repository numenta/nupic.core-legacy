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

#include "gtest/gtest.h"

#include <htm/engine/Input.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/NuPIC.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/os/Env.hpp>
#include <htm/os/Path.hpp>
#include <htm/os/Timer.hpp>
#include <htm/types/Exception.hpp>

#include <cmath>   // fabs/abs
#include <cstdlib> // exit
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>  // for max()
#include <sstream>


namespace testing {

using namespace htm;
using std::exception;

static bool verbose = false;
#define VERBOSE if(verbose) std::cerr << "[          ]"



TEST(CppRegionTest, testCppLinkingFanIn) {
  Network net;
  Real64 *buffer1;
  Real64 *buffer2;
  Real64 *buffer3;

  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{count: 64}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{count: 64}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");

  net.link("region1", "region3"); 
  net.link("region2", "region3"); 

  net.initialize();

  VERBOSE << "Dimensions: \n";
  VERBOSE << " TestNode out  - " << region1->getOutputDimensions("bottomUpOut")    <<"\n";
  VERBOSE << " TestNode out  - " << region2->getOutputDimensions("bottomUpOut")<<"\n";
  VERBOSE << " TestNode in   - " << region3->getInputDimensions("bottomUpIn")  <<"\n";

  const Array r1OutputArray = region1->getOutputData("bottomUpOut");
  const Array r2OutputArray = region2->getOutputData("bottomUpOut");
  Array r3InputArray  = region3->getInputData("bottomUpIn");

  VERBOSE << "region1: " << r1OutputArray << std::endl;
  VERBOSE << "region2: " << r2OutputArray << std::endl;
  VERBOSE << "region3: " << r3InputArray << std::endl;

  EXPECT_EQ(r1OutputArray.getCount(), 64u);
  EXPECT_EQ(r2OutputArray.getCount(), 64u);
  EXPECT_EQ(r3InputArray.getCount(), 128u);

  region1->compute();
  region2->compute();
  VERBOSE << "Checking region1 output after first iteration..." << std::endl;
  VERBOSE << r1OutputArray << std::endl;
  buffer1 = (Real64 *)r1OutputArray.getBuffer();
  ASSERT_EQ(buffer1[0], 0);
  for (size_t i = 1; i < r1OutputArray.getCount(); i++) {
    ASSERT_EQ(buffer1[i], (Real64)(i - 1));
  }

  region3->prepareInputs();
  VERBOSE << "Region3 input after first iteration:" << std::endl;
  VERBOSE << r3InputArray << std::endl;
  ASSERT_EQ(r3InputArray.getCount(), 128u);
  ASSERT_EQ(r3InputArray.getType(), NTA_BasicType_Real64);
  buffer1 = (Real64 *)r1OutputArray.getBuffer();
  buffer2 = (Real64 *)r2OutputArray.getBuffer();
  buffer3 = (Real64 *)r3InputArray.getBuffer();
  for (size_t i = 0; i < 64; i++) {
    ASSERT_EQ(buffer3[i], buffer1[i]);
    ASSERT_EQ(buffer3[i + 64], buffer2[i]);
  }
}


TEST(CppRegionTest, testCppLinkingSDR) {
  Network net;

  std::shared_ptr<Region> region1 = net.addRegion("region1", "ScalarSensor", "{dim: [6,1], n: 6, w: 2}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "SPRegion", "{dim: [20,3]}");

  net.link("region1", "region2"); 

  net.initialize();

  region1->setParameterReal64("sensedValue", 0.8);  //Note: default range setting is -1.0 to +1.0
  const Dimensions r1dims = region1->getOutput("encoded")->getDimensions();
  EXPECT_EQ(r1dims.size(), 2u) << " actual dims: " << r1dims.toString();
  EXPECT_EQ(r1dims[0], 6u) << " actual dims: " << r1dims.toString();
  EXPECT_EQ(r1dims[1], 1u) << " actual dims: " << r1dims.toString();


  region1->compute(); 
  VERBOSE << "Checking region1 output after first iteration..." << std::endl;
  const Array r1OutputArray = region1->getOutputData("encoded");
  VERBOSE << r1OutputArray << "\n";
  std::vector<Byte> expected = {0, 0, 0, 0, 1, 1};
  EXPECT_EQ(r1OutputArray, expected);

  region2->prepareInputs();
  region2->compute();

  VERBOSE << "Region 2 input after first iteration:" << std::endl;
  const Array r2InputArray = region2->getInputData("bottomUpIn");
  VERBOSE << r2InputArray << "\n";
  EXPECT_EQ(r2InputArray.getType(), NTA_BasicType_SDR);
  EXPECT_EQ(r2InputArray,  expected);

  VERBOSE << "Region 2 input after first iteration:" << std::endl;
  const Dimensions r2dims = region2->getOutput("bottomUpOut")->getDimensions();
  EXPECT_EQ(r2dims.size(), 2u) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[0], 20u) << " actual dims: " << r2dims.toString(); //match dims of SPRegion constructed above
  EXPECT_EQ(r2dims[1], 3u) << " actual dims: " << r2dims.toString();
  
  const Array r2OutputArray = region2->getOutputData("bottomUpOut");
  EXPECT_EQ(r2OutputArray.getType(), NTA_BasicType_SDR);
  EXPECT_EQ(r2OutputArray.getSDR().dimensions, r2dims)
      << "Expected dimensions on the output to match dimensions on the buffer.";
  VERBOSE << r2OutputArray << "\n";
  SDR exp({20u, 3u});
  exp.setSparse(SDR_sparse_t{
    4, 21, 32, 46
  });
  EXPECT_EQ(r2OutputArray, exp.getDense()) << "got " << r2OutputArray;
}



TEST(CppRegionTest, testYAML) {
  const char *params = "{count: 42, int32Param: 1234, real64Param: 23.1}";
  //  badparams contains a non-existent parameter
  const char *badparams = "{int32Param: 1234, real64Param: 23.1, badParam: 4}";

  Network net;
  std::shared_ptr<Region> level1;
  EXPECT_THROW(net.addRegion("level1", "TestNode", badparams), exception);

  EXPECT_NO_THROW({level1 = net.addRegion("level1", "TestNode", params);});

  net.initialize();

  // check default values
  Real32 r32val = level1->getParameterReal32("real32Param");
  EXPECT_NEAR(r32val, 32.1f, 0.00001) << "r32val = " << r32val << " diff = " << (r32val - 32.1);

  Int64 i64val = level1->getParameterInt64("int64Param");
  EXPECT_EQ(i64val,  64) << "i64val = " << i64val;

  // check values set in region constructor
  Int32 count = level1->getParameterInt32("count");
  EXPECT_EQ(count, 42);
  Int32 ival = level1->getParameterInt32("int32Param");
  EXPECT_EQ(ival, 1234) << "ival = " << ival;
  Real64 rval = level1->getParameterReal64("real64Param");
  EXPECT_NEAR(rval, 23.1, 0.00000000001) << "rval = " << rval;
  // TODO: if we get the real64 param with getParameterInt32
  // it works -- should we flag an error?

  VERBOSE << "Values are correct for all parameters set at region creation" << std::endl;
}


TEST(CppRegionTest, realmain) {
  Network n;

  size_t count1 = n.getRegions().size();
  EXPECT_TRUE(count1 == 0u);
  std::shared_ptr<Region> level1 = n.addRegion("level1", "TestNode", "{count: 2}");


  size_t count = n.getRegions().size();
  EXPECT_TRUE(count == 1u);
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
  Array b(NTA_BasicType_Int64, &v[0], v.size());
  level1->setParameterArray("int64ArrayParam", b);

  // get the value of intArrayParam after the setParameter call.

  // The array 'b' owns its buffer, so we can call releaseBuffer if we
  // want, but the buffer should be reused if we just pass it again.
  // b.releaseBuffer();
  level1->getParameterArray("int64ArrayParam", b);
  EXPECT_EQ(b.getCount(), 5u) << "set size of TestNode::int64ArrayParam is 5";
  buff = (Int64 *)b.getBuffer();
  for (size_t i = 0; i < std::max<size_t>(b.getCount(), 5u); ++i)
    EXPECT_EQ(buff[i], v[i]) << "Invalid value for index " << i;

  n.run(1);

  const Array& output = level1->getOutputData("bottomUpOut");
  Real64 *data_actual = (Real64 *)output.getBuffer();
  size_t size = output.getCount();
  ASSERT_EQ(size, 2u);
  // set the actual output
  data_actual[1] = 54321.0;
}


TEST(CppRegionTest, RegionSerialization) {
	Network n;
	
	std::shared_ptr<Region> r1 = n.addRegion("testnode", "TestNode", "{count: 2}");
	
	std::stringstream ss;
	r1->save(ss);
	
	Region r2(&n);
	r2.load(ss);
	EXPECT_EQ(*r1.get(), r2);

}


} //ns
