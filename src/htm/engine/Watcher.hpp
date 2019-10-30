/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Interface for the Watcher class
 */

#ifndef NTA_WATCHER_HPP
#define NTA_WATCHER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <htm/engine/Output.hpp>

namespace htm {
class ArrayBase;
class Network;
class Region;


enum watcherType { parameter, output };

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
class Watcher {
public:
  Watcher(const std::string fileName);

  // calls flushFile() and closeFile()
  ~Watcher();

  // returns watchID
  UInt32 watchParam(std::string regionName, std::string varName,
                          int nodeIndex = -1, bool sparseOutput = true);

  // returns watchID
  UInt32 watchOutput(std::string regionName, std::string varName,
                           bool sparseOutput = true);

  // callback function that will be called every time network is run
  static void watcherCallback(Network *net, UInt64 iteration, void *dataIn);

  // Attaches Watcher to a network and begins writing
  // information to a file. Call this after adding all watches.
  void attachToNetwork(Network &);

  // Detaches the Watcher from the Network so the callback is no longer called
  void detachFromNetwork(Network &);

  // Closes the Stream.
  void closeFile();

  // Flushes the Stream.
  void flushFile();

private:

    // Contains data specific for each individual parameter
    // to be watched.
    struct watchData {
        UInt32 watchID; // starts at 1
        std::string varName;
        watcherType wType;
        std::shared_ptr<Output> output;
        // Need regionName because we create data structure before
        // we have the actual Network to attach it to.
        std::string regionName;
        std::shared_ptr<Region> region;
        Int64 nodeIndex;
        NTA_BasicType varType;
        std::string nodeName;
        const ArrayBase *array;
        bool isArray;
        bool sparseOutput;
    };

    // Contains all data needed by the callback function.
    struct allData {
        std::ofstream outStream;
        std::string fileName;
        std::vector<watchData> watches;
    };

  typedef std::vector<watchData> allWatchData;

  // private data structure
  allData data_;
};

} // namespace htm

#endif // NTA_WATCHER_HPP
