/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2014, Numenta, Inc.  Unless you have an agreement
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

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <nupic/engine/Network.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/os/Path.hpp>

namespace testing {

using namespace nupic;

TEST(HelloRegionTest, demo) {

  // Write test data
  if (!Path::exists("TestOutputDir"))
    Directory::create("TestOutputDir");
  std::string path = Path::join("TestOutputDir", "Data.csv");

  // enter some arbitrary test data.
  std::vector<std::vector<Real32>> testdata = {{0, 1.5f, 2.5f},
                                               {1, 1.5f, 2.5f},
                                               {2, 1.5f, 2.5f},
                                               {3, 1.5f, 2.5f}};
  size_t data_rows = testdata.size();
  size_t data_cols = testdata[0].size();
  std::ofstream f(path);
  for (auto row : testdata) {
    for (auto ele : row) {
      f << " " << ele;
    }
    f << std::endl;
  }
  f.close();


  // Create network
  Network net;


  std::string params = "{activeOutputCount: "+std::to_string(data_cols)+"}";

  // Add VectorFileSensor region to network
  std::shared_ptr<Region> region =
      net.addRegion("region", "VectorFileSensor", params);

  // Set region dimensions
  Dimensions dims;
  dims.push_back(1);  // 1 means variable size, in this case set by activeOutputCount.
  std::cout << "Setting region dimensions" << dims.toString() << std::endl;
  region->setDimensions(dims);


  // Load data
  // This will read in all data into a vector of vectors.
  std::vector<std::string> loadFileArgs;
  loadFileArgs.push_back("loadFile");
  loadFileArgs.push_back(path);
  loadFileArgs.push_back("2"); //2=file format, space separated elements.
  region->executeCommand(loadFileArgs);

  // Initialize network
  net.initialize();

  // Compute
  region->compute();  // This should fetch the first row into buffer

  // Get output
  const Array outputArray = region->getOutputData("dataOut");
  EXPECT_TRUE(outputArray.getType() == NTA_BasicType_Real32);
  EXPECT_EQ(outputArray.getCount(), testdata[0].size());
  const Real32 *buffer = (const Real32 *)outputArray.getBuffer();
  for (size_t i = 0; i < outputArray.getCount(); i++)
    EXPECT_FLOAT_EQ(buffer[i], testdata[0][i]);
  // At this point we have consumed the first buffer from VectorFileSensor.

  // Serialize
  Network net2;
  {
    std::stringstream ss;
    net.save(ss);
	//std::cout << "Loading from stream. \n";
    //std::cout << ss.str() << std::endl;
    ss.seekg(0);
    net2.load(ss);
  }
  EXPECT_EQ(net, net2) << "Restored network should be the same as original.";

  std::shared_ptr<Region> region2 = net2.getRegion("region");
  const Array outputArray2 = region2->getOutputData("dataOut");

  // fetch the data rows for both networks.
  net.run((int)data_rows-1);
  net2.run((int)data_rows-1);

  // The last row should be currently in the buffer
  const Real32 *buffer2 = (const Real32 *)outputArray2.getBuffer();
  EXPECT_EQ(data_cols, outputArray.getCount());
  ASSERT_EQ(outputArray2.getCount(), outputArray.getCount());
  for (size_t i = 0; i < outputArray2.getCount(); i++) {
    EXPECT_NEAR(buffer[i], testdata[data_rows -1][i], 0.001);
	  EXPECT_NEAR(buffer[i], buffer2[i], 0.001);
	  std::cout << "testdata=" << testdata[data_rows -1][i]
              << ", buffer=" << buffer[i]
              << ", buffer2=" << buffer2[i] << std::endl;
  }

  // Cleanup
  Directory::removeTree("TestOutputDir", true);
}
} //end namespace
