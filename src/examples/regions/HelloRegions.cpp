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

#include <iostream>
#include <sstream>

#include <nupic/engine/Network.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>



using namespace nupic;

int main(int argc, const char * argv[])
{
    // Create network
    Network net = Network();

    // Add VectorFileSensor region to network
    Region* region = net.addRegion("region", "VectorFileSensor", "{activeOutputCount: 1}");

    // Set region dimensions
    Dimensions dims;
    dims.push_back(1);

    std::cout << "Setting region dimensions" << dims.toString() << std::endl;

    region->setDimensions(dims);

    // Load data
    std::string path = Path::makeAbsolute("../../../src/examples/regions/Data.csv");

    std::cout << "Loading data from " << path << std::endl;

    std::vector<std::string> loadFileArgs;
    loadFileArgs.push_back("loadFile");
    loadFileArgs.push_back(path);
    loadFileArgs.push_back("2");

    region->executeCommand(loadFileArgs);

    // Initialize network
    std::cout << "Initializing network" << std::endl;

    net.initialize();

    ArrayRef outputArray = region->getOutputData("dataOut");

    // Compute
    std::cout << "Compute" << std::endl;

    region->compute();

    // Get output
    Real64 *buffer = (Real64*) outputArray.getBuffer();

    for (size_t i = 0; i < outputArray.getCount(); i++)
    {
        std::cout << "  " << i << "    " << buffer[i] << "" << std::endl;
    }

    // Serialize
    Network net2;
    {
      std::stringstream ss;
      net.write(ss);
      net2.read(ss);
    }
    net2.initialize();

    Region* region2 = net2.getRegions().getByName("region");
    region2->executeCommand(loadFileArgs);
    ArrayRef outputArray2 = region2->getOutputData("dataOut");
    Real64 *buffer2 = (Real64*)outputArray2.getBuffer();

    net.run(1);
    net2.run(1);

    NTA_ASSERT(outputArray2.getCount() == outputArray.getCount());
    for (size_t i = 0; i < outputArray.getCount(); i++)
    {
        std::cout << "  " << i << "    " << buffer[i] << "   " << buffer2[i]
                  << std::endl;
    }

    return 0;
}
