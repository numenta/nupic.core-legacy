/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2017, Numenta, Inc.
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
 * Interface for the Network class
 */

#ifndef NTA_NETWORK_HPP
#define NTA_NETWORK_HPP

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <htm/engine/Region.hpp>
#include <htm/engine/Link.hpp>
#include <htm/ntypes/Collection.hpp>

#include <htm/types/Serializable.hpp>
#include <htm/types/Types.hpp>
#include <htm/utils/Log.hpp>

namespace htm {

class Region;
class Dimensions;
class RegisteredRegionImpl;
class Link;

/**
 * Represents an HTM network. A network is a collection of regions.
 *
 * @nosubgrouping
 */
  class Network : public Serializable
{
public:
  /**
   * @name Construction and destruction
   * @{
   */

  /**
   *
   * Create an new Network
   *
   */
  Network();
  Network(const std::string& filename);

  /**
   * Cannot copy or assign a Network object. But can be moved.
   */
  Network(Network&&);  // move is allowed
  Network(const Network&) = delete;
  void operator=(const Network&) = delete;

  /**
   * Destructor.
   *
   * Destruct the network and unregister it from NuPIC:
   *
   * - Uninitialize all regions
   * - Remove all links
   * - Delete the regions themselves
   *
   * @todo Should we document the tear down steps above?
   */
  ~Network();

  /**
   * Initialize all elements of a network so that it can run.
   *
   * @note This can be called after the Network structure has been set and
   * before Network.run(). However, if you don't call it, Network.run() will
   * call it for you. Also sets up various memory buffers etc. once the Network
   *  structure has been finalized.
   */
  void initialize();

  /**
   * @}
   *
   * @name Internal Serialization methods
   * @{
   */
  /**
   *    saveToFile(path)          Open a file and stream to it. (Binary)
   *    save(ostream f [, fmt])   Stream to your stream.  
   *    f << net;                 Output human readable text. 
   *
   *    loadFromFile(path)
   *    load(istream f [, fmt])
   *
   * @path The filename into which to save/load the streamed serialization.
   * @f    The stream with which to save/load the serialization.
   * @fmt  Format: One of following from enum SerializableFormat
   *   BINARY   - A binary format which is the fastest but not portable between platforms (default).
   *   PORTABLE - Another Binary format, not quite as fast but is portable between platforms.
   *   JSON     - Human readable JSON text format. Slow.
   *   XML      - Human readable XML text format. Even slower.
   *
	 * See Serializable base class for more details.
	 */
  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
std::cerr << "Network.save_ar called\n";
    const std::vector<std::shared_ptr<Link>> links = getLinks();
    std::string name = "Network";
    ar(cereal::make_nvp("name", name));
    ar(cereal::make_nvp("iteration", iteration_));
    ar(cereal::make_nvp("Regions", regions_));
    ar(cereal::make_nvp("links", links));
  }
  
  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
std::cerr << "Network.load_ar called\n";
    std::vector<std::shared_ptr<Link>> links;
    std::string name;
    ar(cereal::make_nvp("name", name));  // ignore value
    ar(cereal::make_nvp("iteration", iteration_));
std::cerr << "Network.load_ar regions\n";
    ar(cereal::make_nvp("Regions", regions_));
std::cerr << "Network.load_ar links\n";
    ar(cereal::make_nvp("links", links));

std::cerr << "Network.load_ar post_load\n";
    post_load(links);
  }

  /**
   * @}
   *
   * @name Region and Link operations
   *
   * @{
   */

  /**
   * Create a new region in a network.
   *
   * @param name
   *        Name of the region, Must be unique in the network
   * @param nodeType
   *        Type of node in the region, e.g. "FDRNode"
   * @param nodeParams
   *        A JSON-encoded string specifying writable params
   *
   * @returns A pointer to the newly created Region
   */
  std::shared_ptr<Region> addRegion(const std::string &name,
  					                        const std::string &nodeType,
                                    const std::string &nodeParams);

  /**
    * Add a region in a network from deserialized region
    *
    * @param Region shared_ptr
    *
    * @returns A pointer to the newly created Region
    */
  std::shared_ptr<Region> addRegion(std::shared_ptr<Region>& region);


  /**
   * Removes an existing region from the network.
   *
   * @param name
   *        Name of the Region
   */
  void removeRegion(const std::string &name);

  /**
   * Create a link and add it to the network.
   *
   * @param srcName
   *        Name of the source region
   * @param destName
   *        Name of the destination region
   * @param linkType
   *        Type of the link
   * @param linkParams
   *        Parameters of the link
   * @param srcOutput
   *        Name of the source output
   * @param destInput
   *        Name of the destination input
   * @param propagationDelay
   *            Propagation delay of the link as number of network run
   *            iterations involving the link as input; the delay vectors, if
   *            any, are initially populated with 0's. Defaults to 0=no delay
   */
  std::shared_ptr<Link> link(const std::string &srcName, const std::string &destName,
            const std::string &linkType="", const std::string &linkParams="",
            const std::string &srcOutput = "",
            const std::string &destInput = "",
            const size_t propagationDelay = 0);

  /**
   * Removes a link.
   *
   * @param srcName
   *        Name of the source region
   * @param destName
   *        Name of the destination region
   * @param srcOutputName
   *        Name of the source output
   * @param destInputName
   *        Name of the destination input
   */
  void removeLink(const std::string &srcName, const std::string &destName,
                  const std::string &srcOutputName = "",
                  const std::string &destInputName = "");

  /**
   * @}
   *
   * @name Access to components
   *
   * @{
   */

  /**
   * Get all regions.
   *
   * @returns A Collection of Region objects in the network
   *          Note: this is a copy of the region list.
   */
  const Collection<std::shared_ptr<Region> > getRegions() const;
  std::shared_ptr<Region> getRegion(const std::string& name) const;

  /**
   * Get all links between regions
   *
   * @returns A Collection of Link objects in the network
   */
  std::vector<std::shared_ptr<Link>> getLinks() const;

  /**
   * Set phases for a region.
   *
   * @param name
   *        Name of the region
   * @param phases
   *        A tuple of phases (must be positive integers)
   */
  void setPhases(const std::string &name, std::set<UInt32> &phases);

  /**
   * Get phases for a region.
   *
   * @param name
   *        Name of the region
   *
   * @returns Set of phases for the region
   */
  std::set<UInt32> getPhases(const std::string &name) const;

  /**
   * Get minimum phase for regions in this network. If no regions, then min = 0.
   *
   * @returns Minimum phase
   */
  UInt32 getMinPhase() const;

  /**
   * Get maximum phase for regions in this network. If no regions, then max = 0.
   *
   * @returns Maximum phase
   */
  UInt32 getMaxPhase() const;

  /**
   * Set the minimum enabled phase for this network.
   *
   * @param minPhase Minimum enabled phase
   */
  void setMinEnabledPhase(UInt32 minPhase);

  /**
   * Set the maximum enabled phase for this network.
   *
   * @param minPhase Maximum enabled phase
   */
  void setMaxEnabledPhase(UInt32 minPhase);

  /**
   * Get the minimum enabled phase for this network.
   *
   * @returns Minimum enabled phase for this network
   */
  UInt32 getMinEnabledPhase() const;

  /**
   * Get the maximum enabled phase for this network.
   *
   * @returns Maximum enabled phase for this network
   */
  UInt32 getMaxEnabledPhase() const;

  /**
   * @}
   *
   * @name Running
   *
   * @{
   */

  /**
   * Run the network for the given number of iterations of compute for each
   * Region in the correct order.
   *
   * For each iteration, Region.compute() is called.
   *
   * @param n Number of iterations
   */
  void run(int n);

  /**
   * The type of run callback function.
   *
   * You can attach a callback function to a network, and the callback function
   *  is called after every iteration of run().
   *
   * To attach a callback, just get a reference to the callback
   * collection with getCallbacks() , and add a callback.
   */
  typedef void (*runCallbackFunction)(Network *, UInt64 iteration, void *);

  /**
   * Type definition for a callback item, combines a @c runCallbackFunction and
   * a `void*` pointer to the associated data.
   */
  typedef std::pair<runCallbackFunction, void *> callbackItem;

  /**
   * Get reference to callback Collection.
   *
   * @returns Reference to callback Collection
   */
  Collection<callbackItem> &getCallbacks();

  /**
   * @}
   *
   * @name Profiling
   *
   * @{
   */

  /**
   * Start profiling for all regions of this network.
   */
  void enableProfiling();

  /**
   * Stop profiling for all regions of this network.
   */
  void disableProfiling();

  /**
   * Reset profiling timers for all regions of this network.
   */
  void resetProfiling();
	
	/**
	 * Set one of the debug levels: LogLevel_None = 0, LogLevel_Minimal, LogLevel_Normal, LogLevel_Verbose
   */
	void setLogLevel(LogLevel level) {
	    LogItem::setLogLevel(level);
	}


  /**
   * @}
   */

  /*
   * Adds a region implementation to the RegionImplFactory's list of packages
   *
   * NOTE: Built-in C++ regions are automatically registered by the factory
   *       so this function does not need to be called.
   *
   * NOTE: How does C++ register a custom C++ implemented region?
   *       Allocate a templated wrapper RegisteredRegionImplCpp class
   *       and pass it to this function with the name of the region type.
   *       Network::registerRegion("MyRegion", new RegisteredRegionImplCpp<MyRegion>());
   *   
   * NOTE: How does Python register a .py implemented region?
   *       Python code should call Network.registerPyRegion(module, className).
   *       The python bindings will actually call the static function
   *       htm::RegisteredRegionImplPy::registerPyRegion(module, className);
   *       which will register the C++ class PyBindRegion as the stand-in for the 
   *       python implementation.
   */
  static void registerRegion(const std::string name, RegisteredRegionImpl *wrapper);
  /*
   * Removes a region implementation from the RegionImplFactory's list of packages
   */
  static void unregisterRegion(const std::string name);

  /*
   * Removes all region registrations in RegionImplFactory.
   * Used in unit tests to setup for next test.
   */
  static void cleanup();

  bool operator==(const Network &other) const;
  inline bool operator!=(const Network &other) const {
    return !operator==(other);
  }

  friend std::ostream &operator<<(std::ostream &, const Network &);

private:
  // Both constructors use this common initialization method
  void commonInit();


  // perform actions after serialization load
  void post_load();
  void post_load(std::vector<std::shared_ptr<Link>>& links);

  // internal method using region pointer instead of name
  void setPhases_(Region *r, std::set<UInt32> &phases);

  // default phase assignment for a new region
  void setDefaultPhase_(Region *region);

  // whenever we modify a network or change phase
  // information, we set enabled phases to min/max for
  // the network
  void resetEnabledPhases_();

  bool initialized_;
	
	/**
	 * The list of regions registered with the Network.
	 * Internally this is a map so it is easy to serialize
	 * but externally this is a Collection object so it
	 * retains API compatability.
	 */
  std::map<std::string, std::shared_ptr<Region>> regions_;

  UInt32 minEnabledPhase_;
  UInt32 maxEnabledPhase_;

  // This is main data structure used to choreograph
  // network computation
  std::vector<std::set<Region *> > phaseInfo_;

  // we invoke these callbacks at every iteration
  Collection<callbackItem> callbacks_;

  // number of elapsed iterations
  UInt64 iteration_;
};

} // namespace htm

#endif // NTA_NETWORK_HPP
