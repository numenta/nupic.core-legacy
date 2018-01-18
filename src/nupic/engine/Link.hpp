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

#include <boost/circular_buffer.hpp>

#include <nupic/engine/LinkPolicy.hpp>
#include <nupic/engine/Input.hpp> // needed for splitter map
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/proto/LinkProto.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>

namespace nupic
{

  class Output;
  class Input;

  /**
   *
   * Represents a link between regions in a Network.
   *
   * @nosubgrouping
   *
   */
  class Link : public Serializable<LinkProto>
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
     * 4. initialize -- sets the offset in the destination Input (not known earlier)
     *
     * De-serializing is the same as phase 1.
     *
     * In phase 3, NuPIC will set and/or get source and/or destination
     * dimensions until both are set. Normally we will only set the src dimensions,
     * and the dest dimensions will be induced. It is possible to go the other
     * way, though.
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
     *            The type of the link
     * @param linkParams
     *            The parameters of the link
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
    Link(const std::string& linkType, const std::string& linkParams,
         const std::string& srcRegionName, const std::string& destRegionName,
         const std::string& srcOutputName="",
         const std::string& destInputName="",
         const size_t propagationDelay=0);

    /**
     * De-serialization use case. Creates a "blank" link. The caller must follow
     * up with Link::read and Link::connectToNetwork
     *
     * @param proto
     *            LinkProto::Reader
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
    void connectToNetwork(Output* src, Input* dest);

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
    Link(const std::string& linkType, const std::string& linkParams,
         Output* srcOutput, Input* destInput, size_t propagationDelay=0);

    /**
     * Initialization Phase 3: set the Dimensions for the source Output, and
     * induce the Dimensions for the destination Input .
     *
     *
     * @param dims
     *         The Dimensions for the source Output
     */
    void setSrcDimensions(Dimensions& dims);

    /**
     * Initialization Phase 3: Set the Dimensions for the destination Input, and
     * induce the Dimensions for the source Output .
     *
     * @param dims
     *         The Dimensions for the destination Input
     */
    void setDestDimensions(Dimensions& dims);

    /**
     * Initialization Phase 4: sets the offset in the destination Input .
     *
     * @param destinationOffset
     *            The offset in the destination Input, i.e. TODO
     *
     */
    void initialize(size_t destinationOffset);

    /**
     * Destructor
     */
    ~Link();

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
    const Dimensions& getSrcDimensions() const;

    /**
     * Get the Dimensions for the destination Input .
     *
     * @returns
     *         The Dimensions for the destination Input
     */
    const Dimensions& getDestDimensions() const;

    /**
     * Get the type of the link.
     *
     * @returns
     *         The type of the link
     */
    const std::string& getLinkType() const;

    /**
     * Get the parameters of the link.
     *
     * @returns
     *         The parameters of the link
     */
    const std::string& getLinkParams() const;

    /**
     * Get the name of the source Region
     *
     * @returns
     *         The name of the source Region
     */
    const std::string& getSrcRegionName() const;

    /**
     * Get the name of the source Output.
     *
     * @returns
     *         The name of the source Output
     */
    const std::string& getSrcOutputName() const;

    /**
     * Get the name of the destination Region.
     *
     * @returns
     *         The name of the destination Region
     *
     */
    const std::string& getDestRegionName() const;

    /**
     * Get the name of the destination Input.
     *
     * @returns
     *         The name of the destination Input
     */
    const std::string& getDestInputName() const;

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
    Output& getSrc() const;

    /**
     *
     * Get the destination Input of the link.
     *
     * @returns
     *         The destination Input of the link
     */
    Input& getDest() const;

    /**
     * Copy data from source to destination.
     *
     * Nodes request input data from their input objects. The input objects,
     * in turn, request links to copy data into the inputs.
     *
     * @note This method must be called on a fully initialized link(all 4 phases).
     *
     */
    void
    compute();

    /**
     * Build a splitter map from the link.
     *
     * @param[out] splitter
     *            The built SplitterMap
     *
     * A splitter map is a matrix that maps the full input
     * of a region to the inputs of individual nodes within
     * the region.
     * A splitter map "sm" is declared as:
     *
     *     vector< vector<size_t> > sm;
     *
     *     sm.length() == number of nodes
     *
     * `sm[i]` is a "sparse vector" used to gather the input
     * for node i. `sm[i].size()` is the size (in elements) of
     * the input for node i.
     *
     * `sm[i]` gathers the inputs as follows:
     *
     *     T *regionInput; // input buffer for the whole region
     *     T *nodeInput; // pre-allocated
     *     for (size_t elem = 0; elem < sm[i].size; elem++)
     *        nodeInput[elem] = regionInput[sm[i][elem]];
     *
     * The offset specified by `sm[i][j]` is in units of elements.
     * To get byte offsets, you'd multiply by the size of an input/output
     * element.
     *
     * An input to a region may come from several links.
     * Each link contributes a contiguous block of the region input
     * starting from a certain offset. The splitter map indices are
     * with respect to the full region input, not the partial region
     * input contributed by this link, so the destinationOffset for this
     * link is included in each of the splitter map entries.
     *
     * Finally, the API is designed so that each link associated with
     * an input can contribute its portion to a full splitter map.
     * Thus the splitter map is an input-output parameter. This method
     * appends data to each row of the splitter map, assuming that
     * existing data in the splitter map comes from other links.
     *
     * For region-level inputs, a splitter map has just a single row.
     *
     * ### Splitter map ownership
     *
     * The splitter map is owned by the containing Input. Each Link
     * in the input contributes a portion to the splitter map, through
     * the buildSplitterMap method.
     *
     */
    void
    buildSplitterMap(Input::SplitterMap& splitter);

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
    friend std::ostream& operator<<(std::ostream& f, const Link& link);

    using Serializable::write;
    void write(LinkProto::Builder& proto) const;

    using Serializable::read;
    void read(LinkProto::Reader& proto);

  private:
    // common initialization for the two constructors.
    void commonConstructorInit_(const std::string& linkType,
                                const std::string& linkParams,
                                const std::string& srcRegionName,
                                const std::string& destRegionName,
                                const std::string& srcOutputName,
                                const std::string& destInputName,
                                const size_t propagationDelay);

    void initPropagationDelayBuffer_(size_t propagationDelay,
                                     NTA_BasicType dataElementType,
                                     size_t dataElementCount);

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

    // We store the values given to use. Use these for
    // serialization instead of serializing the LinkPolicy
    // itself.
    std::string linkType_;
    std::string linkParams_;

    LinkPolicy *impl_;

    Output *src_;
    Input *dest_;

    // Each link contributes a contiguous chunk of the destination
    // input. The link needs to know its offset within the destination
    // input. This value is set at initialization time.
    size_t destOffset_;

    // TODO: These are currently unused. Situations where we need them
    // are rare. Would they make more sense as link policy params?
    // Will also need a link getDestinationSize method since
    // the amount of data contributed by this link to the destination input
    // may not equal the size of the source output.
    size_t srcOffset_;
    size_t srcSize_;

    // Circular buffer for delayed source data buffering
    boost::circular_buffer<Array> srcBuffer_;
    // Number of delay slots
    size_t propagationDelay_;

    // link must be initialized before it can compute()
    bool initialized_;

  };


} // namespace nupic


#endif // NTA_LINK_HPP
