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

#include "gtest/gtest.h"
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Network.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/regions/TestNode.hpp>

static bool verbose = false;
#define VERBOSE                                                                \
  if (verbose)                                                                 \
  std::cerr << "[          ]"

using namespace nupic;

TEST(InputTest, BasicNetworkConstruction) {
  Network net;
  std::shared_ptr<Region> r1 = net.addRegion("r1", "TestNode", "");
  std::shared_ptr<Region> r2 = net.addRegion("r2", "TestNode", "");

  // Test constructor
  Input x(r1.get(), "X", NTA_BasicType_Int32);
  Input y(r2.get(), "Y", NTA_BasicType_Byte);
  EXPECT_THROW(Input i(r1.get(), "I", (NTA_BasicType)(NTA_BasicType_Last + 1)),
               std::exception);

  // test getRegion()
  ASSERT_EQ(r1.get(), x.getRegion());
  ASSERT_EQ(r2.get(), y.getRegion());

  // test isInitialized()
  ASSERT_TRUE(!x.isInitialized());
  ASSERT_TRUE(!y.isInitialized());

  Dimensions d1;
  d1.push_back(8);
  d1.push_back(4);
  r1->setDimensions(d1);
  Dimensions d2;
  d2.push_back(4);
  d2.push_back(2);
  r2->setDimensions(d2);
  net.link("r1", "r2");

  x.initialize();
  y.initialize();

  // test getData()
  const ArrayBase *pa = &(y.getData());
  ASSERT_EQ(0u, pa->getCount());
  Real64 *buf = (Real64 *)(pa->getBuffer());
}


TEST(InputTest, LinkTwoRegionsOneInput1Dmatch) {
  Network net;
  VERBOSE << "Testing [4] + [4] = [8]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{dim: [4]}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{dim: [4]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");
  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();
  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {8};
  Dimensions d3 = region3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0 };
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}

TEST(InputTest, LinkTwoRegionsOneInput1Dnomatch) {
  Network net;
  VERBOSE << "Testing [4] + [3] = [7]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{dim: [4]}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{dim: [3]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");
  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();
  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {7};
  Dimensions d3 = region3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0 };
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}

TEST(InputTest, LinkTwoRegionsOneInput4X3) {
  Network net;
  VERBOSE << "Testing [4,2] + [4] = [4,3]\n";
  std::shared_ptr<Region> region1 =
      net.addRegion("region1", "TestNode", "{dim: [4,2]}");
  std::shared_ptr<Region> region2 =
      net.addRegion("region2", "TestNode", "{dim: [4]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");
  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();
  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {4, 3};
  Dimensions d3 = region3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {1.0, 0.0, 1.0, 2.0, 
                                      1.0, 1.0, 2.0, 3.0,
                                      1.0, 0.0, 1.0, 2.0};
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}

TEST(InputTest, LinkTwoRegionsOneInput4X4) {
  Network net;
  VERBOSE << "Testing [4,2] + [4,2] = [4,4]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");

  Dimensions d1;
  d1.push_back(4);
  d1.push_back(2);
  region1->setDimensions(d1);
  region2->setDimensions(d1);

  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();

  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {4, 4};
  Dimensions d3 = region3->getDimensions();
  VERBOSE << "region3 region dims: " << d3 << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  VERBOSE << "region3 input dims: " << in3->getDimensions() << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0,
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0};
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}


TEST(InputTest, LinkTwoRegionsOneInput3D1) {
  Network net;
  VERBOSE << "Testing [4,2] + [4,2,1] = [4,4,1]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{dim: [4,2]}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{dim: [4,2,1]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");

  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();

  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {4, 4};
  Dimensions d3 = region3->getDimensions();
  VERBOSE << "region3 region dims: " << d3 << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  VERBOSE << "region3 input dims: " << in3->getDimensions() << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0,
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0};
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}


TEST(InputTest, LinkTwoRegionsOneInput3D2) {
  Network net;
  VERBOSE << "Testing [4,2] + [4,2,2] = [4,4,1]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{dim: [4,2]}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{dim: [4,2,2]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");

  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();

  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {4, 2, 3};
  Dimensions d3 = region3->getDimensions();
  VERBOSE << "region3 region dims: " << d3 << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  VERBOSE << "region3 input dims: " << in3->getDimensions() << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0,
      1.0, 0.0, 1.0, 2.0, 
      1.0, 1.0, 2.0, 3.0,
      1.0, 2.0, 3.0, 4.0, 
      1.0, 3.0, 4.0, 5.0};
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}

TEST(InputTest, LinkTwoRegionsOneInputFlatten) {
  Network net;
  VERBOSE << "Testing [4,2] + [3,2] = [14]\n";
  std::shared_ptr<Region> region1 = net.addRegion("region1", "TestNode", "{dim: [4,2]}");
  std::shared_ptr<Region> region2 = net.addRegion("region2", "TestNode", "{dim: [3,2]}");
  std::shared_ptr<Region> region3 = net.addRegion("region3", "TestNode", "");

  net.link("region1", "region3");
  net.link("region2", "region3");

  net.initialize();

  VERBOSE << "region1 output dims: " << region1->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region2 output dims: " << region2->getOutputDimensions("bottomUpOut") << "\n";
  VERBOSE << "region3 region dims: " << region3->getDimensions() << "\n";
  VERBOSE << "region3 input  dims: " << region3->getInputDimensions("bottomUpIn") << "\n";

  Dimensions expected = {14};
  Dimensions d3 = region3->getDimensions();
  VERBOSE << "region3 region dims: " << d3 << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 region dimensions " << expected;

  Input *in3 = region3->getInput("bottomUpIn");
  d3 = in3->getDimensions();
  VERBOSE << "region3 input dims: " << in3->getDimensions() << "\n";
  EXPECT_EQ(d3, expected) << "Expected region3 input dimensions " << expected;

  net.run(2);

  // test getData()
  std::vector<Real64> expectedData = {1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0,
                                      3.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0};
  const Array *pa = &(in3->getData());
  VERBOSE << "region3 input data: " << *pa << std::endl;
  ASSERT_EQ(expectedData.size(), pa->getCount());
  ASSERT_EQ(expectedData, pa->asVector<Real64>());
}
