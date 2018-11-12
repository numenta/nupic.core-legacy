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

#include <nupic/engine/Network.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/os/Path.hpp>

namespace testing {

using namespace nupic;

TEST(HelloRegionTest, demo) {
  // Create network
  Network net;

  // Add VectorFileSensor region to network
  Region *region =
      net.addRegion("region", "VectorFileSensor", "{activeOutputCount: 1}");

  // Set region dimensions
  Dimensions dims;
  dims.push_back(1);

  std::cout << "Setting region dimensions" << dims.toString() << std::endl;

  region->setDimensions(dims);

  // Load data
  const size_t DATA_SIZE = 4; //Data is only 4 rows
  std::string path =
      Path::makeAbsolute("../data/Data.csv"); //FIXME use path relative to CMake's nupic root

  std::cout << "Loading data from " << path << std::endl;

  std::vector<std::string> loadFileArgs;
  loadFileArgs.push_back("loadFile");
  loadFileArgs.push_back(path);
  loadFileArgs.push_back("2"); //2=file format

  region->executeCommand(loadFileArgs);

  // Initialize network
  std::cout << "Initializing network" << std::endl;

  net.initialize();

  ArrayRef outputArray = region->getOutputData("dataOut");

  // Compute
  std::cout << "Compute" << std::endl;

  region->compute();

  // Get output
  const Real64 *buffer = (const Real64 *)outputArray.getBuffer();
  for (size_t i = 0; i < outputArray.getCount(); i++) {
//    EXPECT_FLOAT_EQ(buffer[0], 0); //TODO add test, which values should be here? 
    std::cout << "  " << i << "    " << buffer[i] << "" << std::endl;
  }


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
//  EXPECT_EQ(net, net2);

  Region *region2 = net2.getRegions().getByName("region"); //TODO add more checks and asserts here
  region2->executeCommand(loadFileArgs);
  ArrayRef outputArray2 = region2->getOutputData("dataOut");
  const Real64 *buffer2 = (const Real64 *)outputArray2.getBuffer();

  net.run(DATA_SIZE);
  net2.run(DATA_SIZE);

  ASSERT_EQ(outputArray2.getCount(), outputArray.getCount());
  for (size_t i = 0; i < sizeof &buffer / sizeof &buffer[0]; i++) { //TODO how output all values generated during run(4) ?? 
	  EXPECT_FLOAT_EQ(buffer[i], buffer2[i]);
	  std::cout << " buffer " << buffer[i] << " buffer2: " << buffer2[i] << std::endl;
  }
}
} //end namespace
