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
 * Interface for the Watcher class
 */

#ifndef NTA_WATCHER_HPP
#define NTA_WATCHER_HPP

#include <string>
#include <vector>

#include <nupic/types/ptr_types.hpp>
#include <nupic/engine/Output.hpp>

namespace nupic
{
  class ArrayBase;
  class Network;
  class Region;
  class OFStream;

  enum watcherType
  {
    parameter,
    output
  };
  
  //Contains data specific for each individual parameter
  //to be watched.
  struct watchData
  {
    std::size_t watchID; //starts at 1
    std::string varName;
    watcherType wType;
    Output* output;
    //Need regionName because we create data structure before
    //we have the actual Network to attach it to.
    std::string regionName;
    Region_Ptr_t region;
    Int64 nodeIndex;
    NTA_BasicType varType;
    std::string nodeName;
    const ArrayBase * array;
    bool isArray;
    bool sparseOutput;
  };

  //Contains all data needed by the callback function.
  struct allData
  {
    std::ofstream outStream;
    std::string fileName;
    std::vector<watchData> watches;
  };

  /*
   * Writes the values of parameters and outputs to a file after each
   * iteration of the network.
   *
   * Sample usage:
   *
   * Network net;
   * ...
   * ...
   * 
   * Watcher w("fileName");
   * w.watchParam("regionName", "paramName");
   * w.watchParam("regionName", "paramName", nodeIndex);
   * w.watchOutput("regionName", "bottomUpOut");
   * w.attachToNetwork(net);
   * 
   * net.run();
   *
   * w.detachFromNetwork(net);
   */
  class Watcher
  {
  public:
    Watcher(const std::string fileName);

    //calls flushFile() and closeFile()
    ~Watcher();

    //returns watchID
    auto
        watchParam(std::string regionName,
            std::string varName,
            int nodeIndex = -1,
            bool sparseOutput = true)
    {
        watchData watch;
        watch.varName = varName;
        watch.wType = parameter;
        watch.regionName = regionName;
        watch.nodeIndex = nodeIndex;
        watch.sparseOutput = sparseOutput;
        watch.watchID = data_.watches.size() + 1;
        data_.watches.push_back(watch);
        return watch.watchID;
    }

    //returns watchID
    auto
        watchOutput(std::string regionName,
            std::string varName,
            bool sparseOutput = true)
    {
        watchData watch;
        watch.varName = varName;
        watch.wType = output;
        watch.regionName = regionName;
        watch.nodeIndex = -1;
        watch.isArray = false;
        watch.sparseOutput = sparseOutput;
        watch.watchID = data_.watches.size() + 1;
        data_.watches.push_back(watch);
        return watch.watchID;
    }

    //callback function that will be called every time network is run
    static void
    watcherCallback(Network* net, UInt64 iteration, void* dataIn);
    
    //Attaches Watcher to a network and begins writing
    //information to a file. Call this after adding all watches.
    void
    attachToNetwork(Network&);

    //Detaches the Watcher from the Network so the callback is no longer called
    void
    detachFromNetwork(Network&);

    //Closes the OFStream.
    void
    closeFile();
    
    //Flushes the OFStream.
    void
    flushFile();

  private:
    typedef std::vector<watchData> allWatchData;
    
    //private data structure
    allData data_;
  };

} //namespace nupic

#endif // NTA_WATCHER_HPP
