/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Interface for the Watcher class
 */

#ifndef NTA_WATCHER_HPP
#define NTA_WATCHER_HPP

#include <string>
#include <vector>

#include <nupic/engine/Output.hpp>

namespace nupic {
class ArrayBase;
class Network;
class Region;
class OFStream;

enum watcherType { parameter, output };

// Contains data specific for each individual parameter
// to be watched.
struct watchData {
  unsigned int watchID; // starts at 1
  std::string varName;
  watcherType wType;
  Output *output;
  // Need regionName because we create data structure before
  // we have the actual Network to attach it to.
  std::string regionName;
  Region *region;
  Int64 nodeIndex;
  NTA_BasicType varType;
  std::string nodeName;
  const ArrayBase *array;
  bool isArray;
  bool sparseOutput;
};

// Contains all data needed by the callback function.
struct allData {
  OFStream *outStream;
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
class Watcher {
public:
  Watcher(const std::string fileName);

  // calls flushFile() and closeFile()
  ~Watcher();

  // returns watchID
  unsigned int watchParam(std::string regionName, std::string varName,
                          int nodeIndex = -1, bool sparseOutput = true);

  // returns watchID
  unsigned int watchOutput(std::string regionName, std::string varName,
                           bool sparseOutput = true);

  // callback function that will be called every time network is run
  static void watcherCallback(Network *net, UInt64 iteration, void *dataIn);

  // Attaches Watcher to a network and begins writing
  // information to a file. Call this after adding all watches.
  void attachToNetwork(Network &);

  // Detaches the Watcher from the Network so the callback is no longer called
  void detachFromNetwork(Network &);

  // Closes the OFStream.
  void closeFile();

  // Flushes the OFStream.
  void flushFile();

private:
  typedef std::vector<watchData> allWatchData;

  // private data structure
  allData data_;
};

} // namespace nupic

#endif // NTA_WATCHER_HPP
