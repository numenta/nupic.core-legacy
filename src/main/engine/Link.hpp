/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
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
#include <nta/types/types.hpp>
#include <nta/ntypes/Dimensions.hpp>
#include <nta/engine/LinkPolicy.hpp>
#include <nta/engine/Input.hpp> // needed for splitter map 

namespace nta
{

  class Output;
  class Input;

  /**
   * 
   * Represents a link between regions in a Network . TODO 
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
     * Initialization Phase 1: setting parameter of the link.
     *
     * @param linkType TODO: document
     * @param linkParams TODO: document
     * @param srcRegionName TODO: document
     * @param destRegionName TODO: document
     * @param srcOutputName TODO: document
     * @param destInputName TODO: document
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
         const std::string& srcOutputName="", const std::string& destInputName="");

    /**
     * Initialization Phase 2: connecting inputs/outputs to 
     * the Network.
     * 
     * @param src
     *            The source of the link, an Output
     * @param dest
     *            The destination of the link, an Input
     */
    void connectToNetwork(Output* src, Input* dest);

    /*
     * Initialization Phase 1 and 2.
     *
     * @param linkType TODO: document
     * @param linkParams TODO: document
     * @param srcOutput TODO: document
     * @param destInput TODO: document
     */
    Link(const std::string& linkType, const std::string& linkParams, 
         Output* srcOutput, Input* destInput);

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
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getLinkType() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getLinkParams() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getSrcRegionName() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getSrcOutputName() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getDestRegionName() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    const std::string& getDestInputName() const;

    /**
     * @}
     *
     * @name TODO
     *
     * @{
     */

    // The methods below only work on connected links (after phase 2)

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    Output& getSrc() const;

    /** 
     * TODO: document 
     * @returns TODO: document
     */
    Input& getDest() const;

    /**
     * Copy data from source to destination.
     * 
     * Nodes request input data from their input objects. 
     * 
     * The input objects, in turn, request links to copy data into the inputs.
     *
     * @note This method must be called on a fully initialized link(all 4 phases).
     * 
     */
    void
    compute();

    /**
     * Get the size of the input contributed by this link for a single node.  
     * 
     * @param nodeIndex TODO: document
     * 
     * @returns
     *         The size of the input contributed by this link for a single node. 
     *
     * @todo index=-1 for region-level input?
     */
    size_t
    getNodeInputSize(size_t nodeIndex);

    /**
     * Tells whether the Input is contiguous.
     *
     * @returns
     *         Whether the Input is contiguous, i.e. TODO
     * 
     * If the input for a particular node is a contiguous subset
     * of the src output, then the splitter map is overkill, and 
     * all we need to know is the offset/size (per node)
     * Returns true if and only if the input for each node
     * is a contiguous chunk of the input buffer. 
     * 
     * @todo not implemented;  necessary?
     */
    bool
    isInputContiguous();

    /**
     * Locate the contiguous input for a node. 
     * 
     * This method is used only if the input is contiguous
     * 
     * @todo not implemented;  necessary?
     *
     * @param nodeIndex TODO: document
     * @returns TODO: document
     */
    size_t
    getInputOffset(size_t nodeIndex);

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

    /**
     * @}
     */


  private:
    // common initialization for the two constructors. 
    void commonConstructorInit_(const std::string& linkType, 
                                const std::string& linkParams,
                                const std::string& srcRegionName, 
                                const std::string& destRegionName, 
                                const std::string& srcOutputName, 
                                const std::string& destInputName);

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

    // link must be initialized before it can compute()
    bool initialized_;

  };


} // namespace nta


#endif // NTA_LINK_HPP
