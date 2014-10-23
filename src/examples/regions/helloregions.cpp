/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

#include <iostream>
#include <nta/engine/Network.hpp>
#include <nta/engine/Region.hpp>
#include <nta/ntypes/Dimensions.hpp>
#include <nta/ntypes/ArrayRef.hpp>
#include <nta/os/Path.hpp>

using namespace nta;

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
    std::string path = Path::makeAbsolute("../../../src/examples/regions/data.csv");

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

    return 0;
}
