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

#include <iostream>
#include <string>

#include <nupic/engine/Network.hpp>
#include <nupic/engine/NetworkFactory.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/FStream.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/LoggingException.hpp>
#include <nupic/ntypes/Collection.hpp>
#include <yaml-cpp/yaml.h>


namespace nupic
{
  class Input;
  class Output;

  Network NetworkFactory::createNetwork(const std::string& path)
  { 
    std::string fullPath = Path::normalize(Path::makeAbsolute(path));
    if (! Path::exists(fullPath)) 
    {
      NTA_THROW << "Path " << fullPath << " does not exist";
    }
    std::ifstream f(fullPath.c_str());
    YAML::Parser parser(f);
    return createNetworkFromYAML(parser);
  }

   
  Network NetworkFactory::createNetworkFromYAML(YAML::Parser& parser)
  { 
    YAML::Node doc;
    bool success = parser.GetNextDocument(doc);
    if (!success) 
    {
      NTA_THROW << "Unable to parse YAML document in the specified path.";
    } 

    if (doc.Type() != YAML::NodeType::Map)
    {
      NTA_THROW << "Invalid network structure file -- does not contain a map";
    }
      
    // Should contain Regions and Links only
    if (doc.size() != 2)
    {
      NTA_THROW << "Invalid network structure file -- contains "
                << doc.size() << " elements when it should contain 2.";
    }

    // Regions
    const YAML::Node *regions = doc.FindValue("Regions");
    const YAML::Node *node;
    if (regions == nullptr)
    {
      NTA_THROW << "Invalid network structure file -- no regions";
    }

    if (regions->Type() != YAML::NodeType::Sequence) 
    {
      NTA_THROW << "Invalid network structure file -- regions element is not a list";
    }

    Network n; // Network to be instantiated by the yaml file.

    for (YAML::Iterator region = regions->begin(); region != regions->end(); region++)
    {
      // Each region is a map -- extract the 3 values in the map
      if ((*region).Type() != YAML::NodeType::Map)
      {
        NTA_THROW << "Invalid network structure file -- bad region (not a map)";
      }

      if ((*region).size() != 3)
      {
        NTA_THROW << "Invalid network structure file -- bad region (wrong size)";
      }

      // 1. name
      node = (*region).FindValue("name");
      if (node == nullptr)
      { 
        NTA_THROW << "Invalid network structure file -- region has no name";
      }

      std::string name;
      *node >> name;

      // 2. nodeType
      node = (*region).FindValue("nodeType");
      if (node == nullptr)
      {  
         NTA_THROW << "Invalid network structure file -- region "
                  << name << " has no node type";
      }
       
      std::string nodeType;
      *node >> nodeType;

      // 3. nodeParams
      node = (*region).FindValue("nodeParams");
      if (node == nullptr)
      { 
        NTA_THROW << "Invalid network structure file -- region"
                  << name << "has no nodeParams";
      }

      std::string nodeParams;
      *node >> nodeParams;
      
      // add the region specifed by a map of 3 strings in the sequence.
      n.addRegion(name, nodeType, nodeParams);
    }

    const YAML::Node *links = doc.FindValue("Links");
    const Collection<Region*> regionList = n.getRegions(); // regions in the network.

    if (links == nullptr)
    {  
       NTA_THROW << "Invalid network structure file -- no links";
    }
     
    if (links->Type() != YAML::NodeType::Sequence)
    { 
      NTA_THROW << "Invalid network structure file -- links element is not a list";
    }
     
    for (YAML::Iterator link = links->begin(); link != links->end(); link++)
    {
      // Each link is a map -- extract the 5 values in the map
      if ((*link).Type() != YAML::NodeType::Map)
      { 
        NTA_THROW << "Invalid network structure file -- bad link (not a map)";
      }

      if ((*link).size() != 6)
      { 
        NTA_THROW << "Invalid network structure file -- bad link (wrong size)";
      }

      // 1. type
      node = (*link).FindValue("type");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have a type";
      }
      std::string linkType;
      *node >> linkType;

      // 2. params
      node = (*link).FindValue("params");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have params";
      }

      std::string params;
      *node >> params;

      // 3. srcRegion (name)
      node = (*link).FindValue("srcRegion");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have a srcRegion";
      }

      std::string srcRegionName;
      *node >> srcRegionName;

      // 4. srcOutput
      node = (*link).FindValue("srcOutput");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have a srcOutput";
      }

      std::string srcOutputName;
      *node >> srcOutputName;

      // 5. destRegion
      node = (*link).FindValue("destRegion");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have a destRegion";
      }

      std::string destRegionName;
      *node >> destRegionName;

      // 6. destInput
      node = (*link).FindValue("destInput");
      if (node == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link does not have a destInput";
      }

      std::string destInputName;
      *node >> destInputName;

      if (!regionList.contains(srcRegionName))
      {
        NTA_THROW << "Invalid network structure file -- link specifies source region '" << srcRegionName << "' but no such region exists";
      }

      Region* srcRegion = regionList.getByName(srcRegionName);

      if (!regionList.contains(destRegionName))
      {
        NTA_THROW << "Invalid network structure file -- link specifies destination region '" << destRegionName << "' but no such region exists";
      }

      Region* destRegion = regionList.getByName(destRegionName);

      Output* srcOutput = srcRegion->getOutput(srcOutputName);
      if (srcOutput == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link specifies source output '" << srcOutputName << "' but no such name exists";
      }

      Input* destInput = destRegion->getInput(destInputName);
      if (destInput == nullptr)
      {
        NTA_THROW << "Invalid network structure file -- link specifies destination input '" << destInputName << "' but no such name exists";
      }

      // Create the link itself

      n.link(srcRegionName, destRegionName, linkType, params, srcOutputName, destInputName);
    }
    return n;
  }
} // namespace nupic
