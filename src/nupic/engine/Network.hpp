/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
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
 * Interface for the Network class
 */

#ifndef NTA_NETWORK_HPP
#define NTA_NETWORK_HPP

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Collection.hpp>

#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>

namespace nupic {

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
   * Create an new Network and register it to NuPIC.
   *
   * @note Creating a Network will auto-initialize NuPIC.
   */
  Network();
  Network(const std::string& filename);

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
     *    saveToFile(path)
     *    save(ostream f)
     *    f << net;
     *          serialize everything into one stream.  This can be
     *          opened to a file or a memory stream but must be binary.
     *
     *    loadFromFile(path)
     *    load(istream f)
     *    f >> net;
     *          restores the streamed Network and all its parts back to
     *          what it was before being serialized.
     *
     * @path The filename into which to save/load the streamed serialization.
     * @f    The stream with which to save/load the serialization.
     *
	 * See Serializable base class for definitions.
	 */
    virtual void save(std::ostream &f) const override;
    virtual void load(std::istream &stream)  override;
	virtual void saveToFile(std::string filePath) const override { Serializable::saveToFile(filePath); }
    virtual void loadFromFile(std::string filePath) override { Serializable::loadFromFile(filePath); }

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
  Region_Ptr_t addRegion(const std::string &name,
  					const std::string &nodeType,
                    const std::string &nodeParams);

    /**
     * Create a new region in a network from serialized region
     *
     * @param stream
     *        opened stream
     * @param name
     *        Name of the region, Must be unique in the network.
     *        If not given, it uses the name it was serialized with.
     *
     * @returns A pointer to the newly created Region
     */
    Region_Ptr_t addRegion( std::istream &stream,
                       std::string name = "");


    /**
     * Create a new region in a network from serialized region
	 * The serialized file must have been created by calling SaveToFile()
	 * directly on the region (not on Network).  It restores just this one region.
	 * The fields dimensions and label are not used but provided for backward
	 * compatability.
	 */
    Region_Ptr_t addRegionFromBundle(const std::string name,
					const std::string nodeType,
					const Dimensions& dimensions,
					const std::string& filename,
					const std::string& label = "");

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
  void link(const std::string &srcName, const std::string &destName,
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
   */
  const Collection<Region_Ptr_t > &getRegions() const;
  Region_Ptr_t getRegion(const std::string& name) const;

  /**
   * Get all links between regions
   *
   * @returns A Collection of Link objects in the network
   */
  Collection<Link_Ptr_t> getLinks();

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
   * @}
   */

  /*
   * Adds a region implementation to the RegionImplFactory's list of packages
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

private:
  // Both constructors use this common initialization method
  void commonInit();




  // internal method using region pointer instead of name
  void setPhases_(Region *r, std::set<UInt32> &phases);

  // default phase assignment for a new region
  void setDefaultPhase_(Region *region);

  // whenever we modify a network or change phase
  // information, we set enabled phases to min/max for
  // the network
  void resetEnabledPhases_();

  bool initialized_;
  Collection<Region_Ptr_t> regions_;

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

} // namespace nupic

#endif // NTA_NETWORK_HPP
