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
 * Interface for the Link class
 */

#ifndef NTA_LINK_HPP
#define NTA_LINK_HPP

#include <string>
#include <deque>

#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/types/Types.hpp>
#include <htm/types/Serializable.hpp>

namespace htm {

class Output;
class Input;

/**
 *
 * Represents a link; a data flow connection between regions in a Network.
 *
 * How to use Links:  The big picture.
 *
 * At Configuration time.
 *
 * An application writer would define a pair of links as in the following
 * example given the declaration of a network and three regions,
 *  | Network net;
 *  | auto region1 = net.addRegion("region1", "TestNode", "{count: 64}");
 *  | auto region2 = net.addRegion("region2", "TestNode", "{dim: [64,2]}");
 *  | auto region3 = net.addRegion("region3", "TestNode", "");
 *
 * We can define links from both region1 and region2 into region3 as follows.
 *  | net.link("region1", "region3");
 *  | net.link("region2", "region3");
 *
 * Since only the region names are given, the links connect from the outputs
 * defined as Default in the Spec of each source regions into the input defined as
 * default in the Spec of the destination region (region3 in this case).
 *
 * If we want to be more specific as to which inputs and outputs on each
 * region are to be used, we could use some of the optional fields.
 *  | net.link("region1", "region3", "", "", "bottomUpOut", "bottomUpIn");
 *  | net.link("region2", "region3", "", "", "bottomUpOut", "bottomUpIn");
 *
 * This is equivalent to the link definitions above because the input and outputs
 * specified are the defaults for the 'TestNode' region used in this example.
 *
 * The application must make sure that the right dimensions are configured.
 * Every input and output must end up with a dimension when initialization
 * is called. This can be done as follows:
 *
 * - Fixed Size
 *   If an input or output has the 'count' field in its Spec set to
 *   a non-zero value, this means the region expects the input or output to be
 *   a fixed size.  In this case the size cannot be manually specified.
 *   This fixed size overrides all other Dimensions on this input or output.
 *
 * - Explicit declaration
 *   Dimensions can be manually defined for an input or output as follows:
 *   |   region1->setInputDimensions("bottomUpIn", dim);
 *   |   region1->setOutputDimensions("bottomUpOut", dim);
 *
 * - Declared for region
 *   Dimensions can be manually defined for a region.  Region dimensions are
 *   not directly tied to an input or an output but rather are for the region
 *   as a whole. The region dimensions can be manually defined as follows:
 *   |   region1->setDimensions(dim);
 *
 *   The 'regionLevel' field in the input or output spec relates it
 *   to the region dimensions.
 *
 *   If an input is specified (either directly or by inheriting from its
 *   connected output) and its 'regionLevel' field is true, then its dimensions
 *   will be propogated to the region dimensions.  It can also flow the other
 *   direction. If the region dimensions is specified and the input and its
 *   connected output are not, the input will inherit the region dimensions
 *   and it will also be propogated to the input's connected output.
 *
 *   If an output is defined with the 'regionLevel' field true and the region
 *   dimension is specified, that output will inherit the region dimensions.
 *   The reverse is also true. An output with dimensions can set the region
 *   dimensions if it has not already been set.

 * - Region Dimensions can also be configured on a region using region parameters
 *   as described below.  The advantage is that it can then be included in the
 *   yaml parameter set that is prepared for the application as a whole.
 *   |   auto region2 = net.addRegion("region2", "TestNode", "{dim: [64,2]}");
 *   The 'dim' parameter is a global parameter that is not specifically defined
 *   for a region in the spec. It can be used with any region to set the region's
 *   dimensions.
 *
 * - Dimensions can be indirectly specified for an input or output by the
 *   region implementation when asked for dimensions during initialization.
 *   So, during initialization, the engine attempts to resolve dimensions
 *   for all inputs and outputs.  If an output does not have a dimension
 *   explicitly defined, it will next ask the associated region impl for a
 *   dimension by calling region->askImplForOutputDimensions(output_name);
 *   The region can override this function and provide anything it wants to
 *   ususally something computed from its parameters.  If it does not
 *   override this function, or returns DONTCARE, the base class will call it
 *   with region->getNodeOutputElementCount(output_name) for a 1D dimension.
 *   If this function is not overridden, the base class will return DONTCARE
 *   which tells the engine to use the region in an attempt to derive the dimensions.
 *   Here is an example:
 *   |   auto region1 = net.addRegion("region1", "TestNode", "{count: 64}");
 *
 *   In this case, the 'count' parameter is something that is defined in the
 *   region Spec for the TestNode region.  Other regions may use different
 *   fields to determine buffer sizes.  The 'TestNode' region uses this
 *   to apply a 1D dimension to its 'bottomUpOut' output when it gets asked
 *   for dimensions during initialization.  OR, it could set the region
 *   dimensions directly and let propogation set the dimensions of the inputs
 *   and outputs.  Note that if the count parameter on the TestNode had a
 *   default value, everything will get configured without the application
 *   implementer needing to specify any dimensions or buffer sizes.
 *
 * At initialization time:
 *
 *   the linking logic will create the links between the regions as defined,
 *   determine the dimensions of all inputs and outputs, and then
 *   create the Array buffers for each input and output which match the type
 *   defined in each region's Spec and consistant with it's dimensions.
 *
 *   The Link logic tries to derive any unspecified dimensions so if an
 *   input still does not have a dimension after checking for direct assignment
 *   of a dimension and asking the region for one, it will look at the other
 *   end of its link and propagate the dimensions of the connected output.
 *   But if it still does not have a dimension it will get it from its region
 *   dimension if it was marked as 'regionLevel' in the spec and also propogate
 *   that to its connected output.
 *
 *   If an output does not have a dimension after checking for direct assignment
 *   of a dimension and asking its region for one, it will then try to propogate
 *   the region dimension if this output was marked as 'regionLevel' in the spec.
 *   If it still does not have a dimension, it tries to get it from its connected
 *   input as indicated above.
 *
 *   As it is propagating dimensions there may be a FanIn condition. This is
 *   where more than one output connects to a single input.  In this case the
 *   buffers of all connected outputs are concatinated into the input's buffer.
 *   If the output's dimensions very only in the upper dimension (slowest moving
 *   indexed when being traversed in C) the the input's dimensions will be the
 *   same except that the upper dimension will be the sum of all of the top
 *   level dimensions from the outputs.  example: [100,4] + [100,6] => [100,10].
 *   If the output dimensions are not consistent then everything is flattened
 *   and the input will have the 1D dimensions which is the total number of
 *   elements.  However, if the input also has been configured with dimensions
 *   it will use that as long as the total number of elements are the same.
 *
 *   Once all dimensions are set, Array buffers are created for each input
 *   and output which match the type specified in its spec and the size is
 *   set by its dimensions. The buffers are zero filled.
 *
 *   If an output is configured with a propagation delay, a delay queue is
 *   created and populated with zero filled buffers.
 *
 * At Runtime:
 *   For each iteration, the engine walks through all regions in phase order.
 *   It first prepares the inputs for a region and then calls compute()
 *   on the region which executes the region's algorithm.
 *
 *   Preparing a input for a region means propagating outputs on the other
 *   end of the link into our inputs on our region. If the buffer types are
 *   not the same on each end of the link, a data conversion takes place during
 *   the propagation.  If the types are the same on both ends of the link and
 *   no propagation delay specified, and it is not a FanIn condition, then
 *   the Array is propogated along the link as a shared_ptr and the actual
 *   data does not require a copy.
 *
 *   After the compute() is complete, the algorithm will have left the output
 *   in the output buffer setup during the initialization.
 *   The cycle repeats.
 *
* @nosubgrouping
 *
 */
class Link : public Serializable
{
public:
  /**
   * @name Initialization
   *
   * @{
   *
   * Links have four-phase initialization.
   *
   * 1. construct with link type, params, names of regions and inputs/outputs
   * 2. wire in to network (setting src and dest Output/Input pointers)
   * 3. set source and destination dimensions
   * 4. initialize -- sets the offset in the destination Input (not known
   * earlier)
   *
   * De-serializing is the same as phase 1.
   *
   * In phase 3, NuPIC will set and/or get source and/or destination
   * dimensions until both are set. Normally we will only set the src
   * dimensions, and the dest dimensions will be induced. It is possible to go
   * the other way, though.
   *
   * The @a linkType and @a linkParams parameters are given to
   * the LinkPolicyFactory to create a link policy
   *
   * @todo Should LinkPolicyFactory be documented?
   *
   */

  /**
   * Initialization Phase 1: setting parameters of the link.
   *
   * @param linkType
   *            The type of the link, normally ""
   * @param linkParams
   *            The parameters of the link, normally ""
   * @param srcRegionName
   *            The name of the source Region
   * @param destRegionName
   *            The name of the destination Region
   * @param srcOutputName
   *            The name of the source Output
   * @param destInputName
   *            The name of the destination Input
   * @param propagationDelay
   *            Propagation delay of the link as number of network run
   *            iterations involving the link as input; the delay vectors, if
   *            any, are initially populated with 0's. Defaults to 0=no delay.
   *            Per design, data on no-delay links is to become available to
   *            destination inputs within the same time step, while data on
   *            delayed links (propagationDelay > 0) is to be updated
   *            "atomically" between time steps.
   *
   * @internal
   *
   * @todo It seems this constructor should be deprecated in favor of the other,
   * which is less redundant. This constructor is being used for unit testing
   * and unit testing links and for deserializing networks.
   *
   * See comments below commonConstructorInit_()
   *
   * @endinternal
   *
   */
  Link(const std::string &linkType, const std::string &linkParams,
       const std::string &srcRegionName, const std::string &destRegionName,
       const std::string &srcOutputName = "",
       const std::string &destInputName = "",
       const size_t propagationDelay = 0);

  /**
   * De-serialization use case. Creates a "blank" link. The caller must follow
   * up with Link::deserialize() and Link::connectToNetwork
   *
   */
  Link();

  friend class Network;


  /**
   * Initialization Phase 2: connecting inputs/outputs to
   * the Network.
   *
   * @param src
   *            The source Output of the link
   * @param dest
   *            The destination Input of the link
   */
  void connectToNetwork(std::shared_ptr<Output> src, std::shared_ptr<Input> dest);

  /*
   * Initialization Phase 1 and 2.
   *
   * @param linkType
   *            The type of the link
   * @param linkParams
   *            The parameters of the link
   * @param srcOutput
   *            The source Output of the link
   * @param destInput
   *            The destination Input of the link
   * @param propagationDelay
   *            Propagation delay of the link as number of network run
   *            iterations involving the link as input; the delay vectors, if
   *            any, are initially populated with 0's. Defaults to 0=no delay
   */
  Link(const std::string &linkType, const std::string &linkParams,
       std::shared_ptr<Output> srcOutput, std::shared_ptr<Input> destInput, size_t propagationDelay = 0);


  /**
   * Initialization Phase 4: sets the offset in the destination Input .
   *
   * @param destinationOffset
   *            The offset in the destination Input, i.e. TODO
   *
   */
  void initialize(size_t destinationOffset, bool is_FanIn);


  /**
   * @}
   *
   * @name Parameter getters of the link
   *
   * @{
   *
   */


  /**
   * Get the type of the link.
   *
   * @returns
   *         The type of the link
   */
  const std::string &getLinkType() const;

  /**
   * Get the parameters of the link.
   *
   * @returns
   *         The parameters of the link
   */
  const std::string &getLinkParams() const;

  /**
   * Get the name of the source Region
   *
   * @returns
   *         The name of the source Region
   */
  const std::string &getSrcRegionName() const;

  /**
   * Get the name of the source Output.
   *
   * @returns
   *         The name of the source Output
   */
  const std::string &getSrcOutputName() const;

  /**
   * Get the name of the destination Region.
   *
   * @returns
   *         The name of the destination Region
   *
   */
  const std::string &getDestRegionName() const;

  /**
   * Get the name of the destination Input.
   *
   * @returns
   *         The name of the destination Input
   */
  const std::string &getDestInputName() const;

  /**
   * Get the propogation Delay.
   *
   * @returns
   *         The propogation Delay.
   */
  size_t getPropagationDelay() const { return propagationDelay_; }

  /**
   * @}
   *
   * @name Misc
   *
   * @{
   */

  // The methods below only work on connected links (after phase 2)

  /**
   *
   * Get a generated name of the link in the form
   * RegName.outName --> RegName.inName for debug logging purposes only.
   */
  std::string getMoniker() const;

  /**
   *
   * Get the source Output of the link.
   *
   * @returns
   *         The source Output of the link
   */
  Output* getSrc() const;

  /**
   *
   * Get the destination Input of the link.
   *
   * @returns
   *         The destination Input of the link
   */
  Input* getDest() const;

  /**
   * Copy data from source to destination.
   *
   * Nodes request input data from their input objects. The input objects,
   * in turn, request links to copy data into the inputs.
   *
   * @note This method must be called on a fully initialized link(all 4 phases).
   *
   */
  void compute();


  /*
   * No-op for links without delay; for delayed links, remove head element of
   * the propagation delay buffer and push back the current value from source.
   *
   * NOTE It's intended that this method be called exactly once on all links
   * within a network at the end of every time step. Network::run calls it
   * automatically on all links at the end of each time step.
   */
  void shiftBufferedData();

  /**
   * Convert the Link to a human-readable string.
   *
   * @returns
   *     The human-readable string describing the Link
   */
  const std::string toString() const;

  void setOffset(size_t count) { destOffset_ = count; }

  /**
   * Display and compare the link.
   *
   * @param f
   *            The output stream being serialized to
   * @param link
   *            The Link being serialized
   */
  friend std::ostream &operator<<(std::ostream &f, const Link &link);
  bool operator==(const Link &o) const;
  bool operator!=(const Link &o) const { return !operator==(o); }

  /**
   * Serialize/Deserialize the link.
   */
  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    std::deque<Array> delay = preSerialize();
    ar(cereal::make_nvp("srcRegionName", srcRegionName_),
       cereal::make_nvp("srcOutputName", srcOutputName_),
       cereal::make_nvp("destRegionName", destRegionName_),
       cereal::make_nvp("destInputName", destInputName_),
       cereal::make_nvp("destOffset", destOffset_),
       cereal::make_nvp("is_FanIn", is_FanIn_),
       cereal::make_nvp("propagationDelay", propagationDelay_),
       cereal::make_nvp("propagationDelayBuffer", delay));
  }
  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(cereal::make_nvp("srcRegionName", srcRegionName_),
       cereal::make_nvp("srcOutputName", srcOutputName_),
       cereal::make_nvp("destRegionName", destRegionName_),
       cereal::make_nvp("destInputName", destInputName_),
       cereal::make_nvp("destOffset", destOffset_),
       cereal::make_nvp("is_FanIn", is_FanIn_),
       cereal::make_nvp("propagationDelay", propagationDelay_),
       cereal::make_nvp("propagationDelayBuffer", propagationDelayBuffer_));
    initialized_ = false;
  }

private:
  // common initialization for the two Link constructors.
  void commonConstructorInit_(const std::string &linkType,
                              const std::string &linkParams,
                              const std::string &srcRegionName,
                              const std::string &destRegionName,
                              const std::string &srcOutputName,
                              const std::string &destInputName,
                              const size_t propagationDelay);

  std::deque<Array> preSerialize() const;


  std::string srcRegionName_;
  std::string destRegionName_;
  std::string srcOutputName_;
  std::string destInputName_;

  // We store the values given to use. No longer used.
  std::string linkType_;
  std::string linkParams_;

  // Note: these must be raw pointers to avoid circular linkages with shared_ptrs.
  Output* src_;
  Input* dest_;

  // Each link contributes a contiguous chunk of the destination
  // input. The link needs to know its offset within the destination
  // input. This value is set at initialization time.
  size_t destOffset_;
  bool is_FanIn_;

  // Queue buffer for delayed source data buffering
  std::deque<Array> propagationDelayBuffer_;
  // Number of delay slots
  size_t propagationDelay_;

  // link must be initialized before it can compute()
  bool initialized_;
};

} // namespace htm

#endif // NTA_LINK_HPP
