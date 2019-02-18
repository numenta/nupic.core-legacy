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
 * Interface for the Link class
 */

#ifndef NTA_LINK_HPP
#define NTA_LINK_HPP

#include <string>
#include <deque>

#include <nupic/engine/Input.hpp> // needed for splitter map
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/types/Types.hpp>

namespace nupic {

class Output;
class Input;

/**
 *
 * Represents a link between regions in a Network.
 *
 * @nosubgrouping
 *
 */
class Link
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

  /**
   * Initialization Phase 2: connecting inputs/outputs to
   * the Network.
   *
   * @param src
   *            The source Output of the link
   * @param dest
   *            The destination Input of the link
   */
  void connectToNetwork(Output *src, Input *dest);

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
       Output *srcOutput, Input *destInput, size_t propagationDelay = 0);

  /**
   * Initialization Phase 3: set the Dimensions for the source Output, and
   * induce the Dimensions for the destination Input .
   *
   *
   * @param dims
   *         The Dimensions for the source Output
   */
  void setSrcDimensions(Dimensions &dims);

  /**
   * Initialization Phase 3: Set the Dimensions for the destination Input, and
   * induce the Dimensions for the source Output .
   *
   * @param dims
   *         The Dimensions for the destination Input
   */
  void setDestDimensions(Dimensions &dims);

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
   * Get the Dimensions for the source Output .
   *
   * @returns
   *         The Dimensions for the source Output
   */
  const Dimensions &getSrcDimensions() const;

  /**
   * Get the Dimensions for the destination Input .
   *
   * @returns
   *         The Dimensions for the destination Input
   */
  const Dimensions &getDestDimensions() const;

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
  Output &getSrc() const;

  /**
   *
   * Get the destination Input of the link.
   *
   * @returns
   *         The destination Input of the link
   */
  Input &getDest() const;

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

  /**
   * Serialize the link.
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
   * Serialize the link using a stream.
   *
   * @param f -- The stream to output to.
   */
  void serialize(std::ostream &f);

  /**
   * Deserialize the link from binary stream.
   *
   * @param f -- the stream to read from
   *
   */
  void deserialize(std::istream &f);

private:
  // common initialization for the two Link constructors.
  void commonConstructorInit_(const std::string &linkType,
                              const std::string &linkParams,
                              const std::string &srcRegionName,
                              const std::string &destRegionName,
                              const std::string &srcOutputName,
                              const std::string &destInputName,
                              const size_t propagationDelay);



  // TODO: The strings with src/dest names are redundant with
  // the src_ and dest_ objects. For unit testing links,
  // and for deserializing networks, we need to be able to create
  // a link object without a network. and for deserializing, we
  // need to be able to instantiate a link before we have instantiated
  // all the regions. (Maybe this isn't true? Re-evaluate when
  // more infrastructure is in place).

  std::string srcRegionName_;
  std::string destRegionName_;
  std::string srcOutputName_;
  std::string destInputName_;

  // We store the values given to use. No longer used.
  std::string linkType_;
  std::string linkParams_;

  // ---
  // The dimensions of the source Region, as specified by a call to
  // setSrcDimensions() or induced by a call to setDestDimensions().
  // ---
  Dimensions srcDimensions_;

  // ---
  // The dimensions of the destination Region, as specified by a call to
  // setDestDimensions() or induced by a call to setSrcDimensions().
  // ---
  Dimensions destDimensions_;

  // ---
  // The amount of elements per Node as specified by a call to
  // setNodeOutputElementCount()
  // ---
  size_t elementCount_;

  Output *src_;
  Input *dest_;

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

} // namespace nupic

#endif // NTA_LINK_HPP
