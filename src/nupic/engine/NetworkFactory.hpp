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

/** @file 
 * Definition of the NetworkFactory API
 */

#include <string>
#include <yaml-cpp/yaml.h>

namespace nupic
{

  class Network; 

  class NetworkFactory
  {
  public:

    NetworkFactory() {};

    ~NetworkFactory() {};

    // Return a Network based on the yaml file passed in. Calls
    // createNetworkFromYaml after passing in the yaml string from 
    // the file specified by path.
    Network* createNetwork(const std::string &path);

    // Create a Network as specified by the YAML string passed in.
    Network* createNetworkFromYAML(const std::string &path);

  private:

  };

} // namespace nupic
