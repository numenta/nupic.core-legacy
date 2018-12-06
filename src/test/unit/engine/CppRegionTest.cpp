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
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp> // memory leak detection
#include <nupic/os/Path.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/types/Exception.hpp>

#include <cmath>   // fabs/abs
#include <cstdlib> // exit
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


namespace testing {

using namespace nupic;
using std::exception;

bool verbose = false;


struct MemoryMonitor {
  MemoryMonitor(const size_t count) {
	    NTA_ASSERT(count > 1 && count < minCount)
    << "Run count of " << count << " specified\n"
    << "When run in leak detection mode, count must be at least "
    << minCount << "\n";

	  OS::getProcessMemoryUsage(initial_vmem, initial_rmem);
  }

  ~MemoryMonitor() {
    if (hasMemoryLeaks()) {
      NTA_DEBUG << "Memory leaks detected. "
                << "Real Memory: " << diff_rmem
                << ", Virtual Memory: " << diff_vmem;
    }
  }

  void update() {
    i++;
    if(i < memoryLeakStartIter) return; //not yet initialized
    if(i % memoryLeakDeltaIterCheck != 0) return;

    OS::getProcessMemoryUsage(current_vmem, current_rmem);

    if (i == memoryLeakStartIter) { //start, re-init mem values
      initial_rmem = current_rmem;
      initial_vmem = current_vmem;
      std::cout << "Memory usage: " << current_vmem << " (virtual) "
                  << current_rmem << " (real) at iteration " << i << std::endl;
      return;
    }
    diff_vmem = current_vmem - initial_vmem;
    diff_rmem = current_rmem - initial_rmem;
  }


  bool hasMemoryLeaks() {
    update();
    return diff_vmem > 0 || diff_rmem > 0;
  }

	private:
  size_t initial_vmem;
  size_t initial_rmem;
  size_t current_vmem;
  size_t current_rmem;
  size_t diff_rmem;
  size_t diff_vmem;
  size_t i = 0; //current iter
    // Start checking memory usage after this many iterations.
#if defined(NTA_OS_WINDOWS)
  // takes longer to settle down on win32
  const static size_t memoryLeakStartIter = 6000;
#else
  const static size_t memoryLeakStartIter = 150;
#endif
  // This determines how frequently we check.
  const size_t memoryLeakDeltaIterCheck = 10;
  const size_t minCount = memoryLeakStartIter + 5 * memoryLeakDeltaIterCheck;
};


void helperCppInputOutputAccess(Region *level1) {
  // --- input/output access for level 1 (C++ TestNode) ---

//  SHOULDFAIL(level1->getOutputData("doesnotexist"));

  // getting access via zero-copy
  std::cout << "Getting output for zero-copy access" << std::endl;
  ArrayRef output = level1->getOutputData("bottomUpOut");
  std::cout << "Element count in bottomUpOut is " << output.getCount() << ""
            << std::endl;
  Real64 *data_actual = (Real64 *)output.getBuffer();
  // set the actual output
  data_actual[12] = 54321;
}



TEST(CppRegionTest, testCppLinkingFanIn) {
  Network net = Network();

  Region_Ptr_t region1 = net.addRegion("region1", "TestNode", "");
  Region_Ptr_t region2 = net.addRegion("region2", "TestNode", "");

  net.link("region1", "region2", "TestFanIn2", ""); //the only change testCppLinking* is here

  std::cout << "Initialize should fail..." << std::endl;
  EXPECT_THROW(net.initialize(), exception);

  std::cout << "Setting region1 dims" << std::endl;
  Dimensions r1dims;
  r1dims.push_back(6);
  r1dims.push_back(4);
  region1->setDimensions(r1dims);

  std::cout << "Initialize should now succeed" << std::endl;
  net.initialize();

  const Dimensions &r2dims = region2->getDimensions();
  EXPECT_EQ(r2dims.size(), 2) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[0], 3) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[1], 2) << " actual dims: " << r2dims.toString();

  EXPECT_THROW(region2->setDimensions(r1dims), exception);

  ArrayRef r1OutputArray = region1->getOutputData("bottomUpOut");

  region1->compute();
  std::cout << "Checking region1 output after first iteration..." << std::endl;
  Real64 *buffer = (Real64 *)r1OutputArray.getBuffer();

  for (size_t i = 0; i < r1OutputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  region2->prepareInputs();
  ArrayRef r2InputArray = region2->getInputData("bottomUpIn");
  std::cout << "Region 2 input after first iteration:" << std::endl;
  Real64 *buffer2 = (Real64 *)r2InputArray.getBuffer();
  EXPECT_TRUE(buffer != buffer2);

  for (size_t i = 0; i < r2InputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  std::cout << "Region 2 input by node" << std::endl;
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
    int row = node / 3;
    int col = node - (row * 3);
    EXPECT_EQ(r2NodeInput.size(), 8);
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
  Network net = Network();

  Region_Ptr_t region1 = net.addRegion("region1", "TestNode", "");
  Region_Ptr_t region2 = net.addRegion("region2", "TestNode", "");

  net.link("region1", "region2", "UniformLink", "{mapping: in, rfSize: [2]}"); //the only change testCppLinking* is here


  std::cout << "Initialize should fail..." << std::endl;
  EXPECT_THROW(net.initialize(), exception);

  std::cout << "Setting region1 dims" << std::endl;
  Dimensions r1dims;
  r1dims.push_back(6);
  r1dims.push_back(4);
  region1->setDimensions(r1dims);

  std::cout << "Initialize should now succeed" << std::endl;
  net.initialize();

  const Dimensions &r2dims = region2->getDimensions();
  EXPECT_EQ(r2dims.size(), 2) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[0], 3) << " actual dims: " << r2dims.toString();
  EXPECT_EQ(r2dims[1], 2) << " actual dims: " << r2dims.toString();

  EXPECT_THROW(region2->setDimensions(r1dims), exception);

  ArrayRef r1OutputArray = region1->getOutputData("bottomUpOut");

  region1->compute();
  std::cout << "Checking region1 output after first iteration..." << std::endl;
  Real64 *buffer = (Real64 *)r1OutputArray.getBuffer();
  for (size_t i = 0; i < r1OutputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  region2->prepareInputs();
  ArrayRef r2InputArray = region2->getInputData("bottomUpIn");
  std::cout << "Region 2 input after first iteration:" << std::endl;
  Real64 *buffer2 = (Real64 *)r2InputArray.getBuffer();
  EXPECT_TRUE(buffer != buffer2);

  for (size_t i = 0; i < r2InputArray.getCount(); i++) {
    if (i % 2 == 0)
      ASSERT_EQ(buffer[i], 0);
    else
      ASSERT_EQ(buffer[i], (i - 1) / 2);
  }

  std::cout << "Region 2 input by node" << std::endl;
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
    int row = node / 3;
    int col = node - (row * 3);
    EXPECT_EQ(r2NodeInput.size(), 8);
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



TEST(CppRegionTest, testYAML) {
  const char *params = "{int32Param: 1234, real64Param: 23.1}";
  //  badparams contains a non-existent parameter
  const char *badparams = "{int32Param: 1234, real64Param: 23.1, badParam: 4}";

  Network net = Network();
  Region_Ptr_t level1;
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

  std::cout
      << "Got the correct values for all parameters set at region creation"
      << std::endl;
}



Network helperRealmain() {
  // verbose == true turns on extra output that is useful for
  // debugging the test (e.g. when the TestNode compute()
  // algorithm changes)

  std::cout << "Creating network..." << std::endl;
  Network n;

  std::cout << "Region count is " << n.getRegions().getCount() << ""
            << std::endl;

  std::cout << "Adding a FDRNode region..." << std::endl;
  Region_Ptr_t level1 = n.addRegion("level1", "TestNode", "");

  std::cout << "Region count is " << n.getRegions().getCount() << ""
            << std::endl;
  std::cout << "Node type: " << level1->getType() << "" << std::endl;
  std::cout << "Nodespec is:\n"
            << level1->getSpec()->toString() << "" << std::endl;

  Int64 val;
  Real64 rval;
  std::string int64Param("int64Param");
  std::string real64Param("real64Param");

  val = level1->getParameterInt64(int64Param);
  rval = level1->getParameterReal64(real64Param);
  std::cout << "level1.int64Param = " << val << "" << std::endl;
  std::cout << "level1.real64Param = " << rval << "" << std::endl;

  val = 20;
  level1->setParameterInt64(int64Param, val);
  val = 0;
  val = level1->getParameterInt64(int64Param);
  std::cout << "level1.int64Param = " << val << " after setting to 20"
            << std::endl;

  rval = 30.1;
  level1->setParameterReal64(real64Param, rval);
  rval = 0.0;
  rval = level1->getParameterReal64(real64Param);
  std::cout << "level1.real64Param = " << rval << " after setting to 30.1"
            << std::endl;

  // --- test getParameterInt64Array ---
  // Array a is not allocated by us. Will be allocated inside getParameter
  Array a(NTA_BasicType_Int64);
  level1->getParameterArray("int64ArrayParam", a);
  std::cout << "level1.int64ArrayParam size = " << a.getCount() << std::endl;
  std::cout << "level1.int64ArrayParam = [ ";
  Int64 *buff = (Int64 *)a.getBuffer();
  for (int i = 0; i < int(a.getCount()); ++i)
    std::cout << buff[i] << " ";
  std::cout << "]" << std::endl;

  // --- test setParameterInt64Array ---
  std::cout << "Setting level1.int64ArrayParam to [ 1 2 3 4 ]" << std::endl;
  std::vector<Int64> v(4);
  for (int i = 0; i < 4; ++i)
    v[i] = i + 1;
  Array newa(NTA_BasicType_Int64, &v[0], v.size());
  level1->setParameterArray("int64ArrayParam", newa);

  // get the value of intArrayParam after the setParameter call.

  // The array a owns its buffer, so we can call releaseBuffer if we
  // want, but the buffer should be reused if we just pass it again.
  // a.releaseBuffer();
  level1->getParameterArray("int64ArrayParam", a);
  std::cout << "level1.int64ArrayParam size = " << a.getCount() << std::endl;
  std::cout << "level1.int64ArrayParam = [ ";
  buff = (Int64 *)a.getBuffer();
  for (int i = 0; i < int(a.getCount()); ++i)
    std::cout << buff[i] << " ";
  std::cout << "]" << std::endl;

  return n;
}


TEST(CppRegionTest, realmain) {
  Network n = helperRealmain();

  // should fail because network has not been initialized
  EXPECT_THROW(n.run(1), exception);

  // should fail because network can't be initialized
  EXPECT_THROW(n.initialize(), exception);

  std::cout << "Setting dimensions of level1..." << std::endl;
  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  Region_Ptr_t level1 = n.getRegion("level1");
  level1->setDimensions(d);

  std::cout << "Initializing again..." << std::endl;
  EXPECT_NO_THROW(n.initialize());

  level1->compute();

  EXPECT_NO_THROW(helperCppInputOutputAccess(level1.get()));
  EXPECT_THROW(level1->getOutputData("doesnotexist"), exception);

//  EXPECT_NO_THROW(testCppLinking("TestFanIn2", ""));  //now called in separate test, but could/should also be called here
//  EXPECT_NO_THROW(testCppLinking("UniformLink", "{mapping: in, rfSize: [2]}"));
//  EXPECT_NO_THROW(testYAML());

}



TEST(DISABLED_CppRegionTest, memLeak) { //FIXME this mem leak test is newly fixed, but catches error -> need to fix code
  /*
   * With an integer argument 'count', runs the same test N times
   * and requires that memory use stay constant -- it can't
   * grow by even one byte.
   */
  const size_t count = 8000;

  MemoryMonitor m(count);
  for (size_t i = 0; i < count; i++) {
	//call main
  Network n = helperRealmain();

  //cannot use EXPECT_THROW in EXPECT_THROW
  std::cout << "Setting dimensions of level1..." << std::endl;
  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  Region_Ptr_t level1 = n.getRegion("level1");
  level1->setDimensions(d);

  std::cout << "Initializing again..." << std::endl;
  n.initialize();
  ASSERT_TRUE(NuPIC::isInitialized()) << "now must be initialized";

  level1->compute();
  helperCppInputOutputAccess(level1.get());
	//end main

      // testExceptionBug();
      // testCppLinking("TestFanIn2","");
      //NuPIC::shutdown();
      // memory leak detection
      // we check even prior to the initial tracking iteration, because the act
      // of checking potentially modifies our memory usage
      EXPECT_FALSE(m.hasMemoryLeaks());
  }

  std::cout << "--- ALL TESTS PASSED ---" << std::endl;
}

} //ns
