/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

#ifndef NTA_NETWORK_FACTORY_HPP
#define NTA_NETWORK_FACTORY_HPP

#include <iosfwd>  // for string
namespace YAML { class Parser; }

namespace nupic
{
  class Network;
  
  /** Factory for creating Network instances from YAML files.
   *
   * @b Description
   * Specify the structure of a Network in YAML file and use
   * this factory to create that network using the Network API.
   */
  class NetworkFactory
  {
  public:

    NetworkFactory() {};

    ~NetworkFactory() {};

    /** Create a Network instance based on the yaml file passed in. 
     *
     *  Calls createNetworkFromYaml after ensuring the path has a yaml extension and
     *  exists.
     *
     *  the file specified by path is expected to be a .yaml file with the outer
     *  most element a map of two elements (Regions and Links).
     *
     *  Region:
     *    - name       : string
     *      nodeType   : string
     *      nodeParams : string
     *    ...
     *  Links:
     *    - srcRegion  : string
     *      destRegion : string
     *      type       : string
     *      params     : string
     *      srcOutput  : string
     *      destInput  : string
     *  ...
     * @param path the .yaml file path
     * @retval A pointer to the Network object specified by the yaml.
     */
    Network createNetwork(const std::string& path);

    /** Internal method to parse the yaml and return the Network Instance.
     *
     * @param parser - parser of the .yaml
     * @retval - a pointer to the Network object instance specified by the yaml.
     */
    Network createNetworkFromYAML(YAML::Parser& p);

  private:

  };

} // namespace nupic

#endif // NTA_NETWORK_FACTORY_HPP
