/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2015, Numenta, Inc.
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
 * Definition of the RegionImpl API
 *
 * A RegionImpl is a node "plugin" that provides most of the
 * implementation of a Region, including algorithms.
 *
 * The RegionImpl class is expected to be subclassed for particular
 * node types (e.g. TestNode, SPRegion, TMRegion, etc) and RegionImpls are 
 * instatiated by the RegionImplFactory.
 *
 * This is a wrapper or interface for an algorithm so that it
 * can be used within the NetworkAPI framework.  
 *
 * A region implementation must subclass this class and implement the following:
 *   a) Implement the static function getSpec() to return a
 *      specification that describes how this region will be handled.
 *      It must describe parameters, inputs, outputs, and commands.
 *   b) Obtain the parameters passed in the Constructor.
 *   c) Implement either askImplForOutputDimensions() -preferred, or
 *      getNodeOutputElementCount() to provide information needed to 
 *      allocate output buffers.
 *   d) implement getParameterXXX() and setParameterXXX() as needed.
 *      Spec parameter access specification dictates which of these
 *      functions need to be implemented for each parameter.
 *   e) Implement executeCommand() for each command defined in the Spec.
 *   f) Implement initialize().  This is normally where its algorithm is
 *      initantiated because parameters and dimensions will have been set.
 *   g) Implement compute().  This is where we execute the algorithm, passing
 *      in the input data and obtaining the output data.
 *   h) Implement serialization using save_ar() and load_ar() functions.
 *
 *
 * Normal Processing flow for a Region implemementation:
 * 1) Registration:
 *    If this is a custom region, C++ users must call
 *       Network.registerRegion(name, RegisteredRegionImpl)
 *    and Python users must call Network.registerPyRegion(module, class)
 *    to register their implementations.  Built-in C++ implementations
 *    are already registered by RegionImplFactory.
 *
 * 2) Region Creation:
 *    When a region is added to the Network with Network.addRegion(name, params),
 *    The Region and its corresponding region implementation will be
 *    instantiated.  When the region implementation's constructors are
 *    called, they are passed the parameters provided in the addRegion() call
 *    merged with the default parameters from the Spec.  The constructor
 *    must pick up these parameters and store locally.
 *
 *    The parameter 'dim' will already be handled and will set the region
 *    level dimensions.
 *
 *    During this call, the Inputs and Output objects are created for 
 *    each input and output specified in the Spec and attached to the
 *    region. The buffers and dimensions are set later.
 *
 * 3) Link Creation:
 *    After regions are created, they may be used in a link() call to 
 *    create a path for data to flow between the regions. 
 *
 * 4) Configuration:
 *    Following region creation, a user may make setParameterXXX() calls
 *    that can modify parameters for a region.  It may also call setDimensions() to 
 *    manually set or override the dimensions. Normally these are not
 *    needed but this provides a way to handle unique situations.
 *
 * 5) Initialize:
 *    Next the user should call Network.initialize(). If this is not
 *    called, the first call to Network.run() will call initialize.
 *
 *    This call will first evaluate the links, determine the dimensions
 *    for all inputs and outputs and create the buffers. If it cannot
 *    determine dimensions it will call askImplForInputDimensions()
 *    and askImplForOutputDimensions() on a region implementation to 
 *    obtain dimensions.  If the region impl did not implement those functions
 *    it will call getNodeInputElementCount() and getNodeOutputElementCount()
 *    to obtain buffer size to use as a dimension. See Link.hpp for a
 *    more complete description of how dimensions are set.
 *
 *    At this point, when all dimensions are determined and all buffers
 *    have been allocated, it will then call initialize() on each region 
 *    implementation.  This is normally when the region's algorithm
 *    object is instatiated and parameters and dimensions are passed in.
 *
 *    Any parameter changes made by setParameterXXX() must be passed
 *    on to the algorithm if allowed and getParameterXXX()  must query 
 *    the algorithm if allowed.
 *
 * 6) Run:
 *    At this point we are ready to execute everything. When Network.run()
 *    is called, the Network class will sequentually call the compute() 
 *    method on each region implementation in the phase order and within 
 *    each phase in the order of region declaration.
 *
 *    This will iterate for as many times as given in the run() argument.
 *    At the beginning of each iteration any callbacks that were configured
 *    during Configuration time will be called allowing the user to sample
 *    data or adjust parameters.
 *
 *    Prior to each region's compute() call, all links are traversed and 
 *    all connected region outputs are copied (or moved) to the corresponding 
 *    region inputs.  A previously executed region's outputs are then
 *    available to a subsequent region, allowing data to cascade through 
 *    the links within the same iteration. Buffering and data type conversions 
 *    are automatically performed as needed based on the data types declared 
 *    in the region Spec for each input and output.
 *
 * 7) Serialization:
 *    Serialization will save the current state of the region and its 
 *    algorithm in such a way that when restored, execution may resume
 *    where it left off.  Input and Output buffers do not need to be
 *    saved because they are handled by the Region class.
 */

#ifndef NTA_REGION_IMPL_HPP
#define NTA_REGION_IMPL_HPP

#include <iostream>
#include <string>
#include <vector>

#include <htm/engine/Output.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Region.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Value.hpp>

namespace htm {

class Spec;
class Region;
class Dimensions;
class Input;
class Output;
class Array;
class NodeSet;

class RegionImpl
{
public:
  // All subclasses must call this constructor from their regular constructor
  RegionImpl(Region *region);

  virtual ~RegionImpl();

  /* ------- Convenience methods  that access region data -------- */

  std::string getType() const;

  std::string getName() const;


  /* ------- Parameter support in the base class. ---------*/
  // The default implementation of all of these methods goes through
  // set/getParameterFromBuffer, which is compatible with NuPIC 1.
  // RegionImpl subclasses may override for higher performance.

  virtual Int32 getParameterInt32(const std::string &name, Int64 index);
  virtual UInt32 getParameterUInt32(const std::string &name, Int64 index);
  virtual Int64 getParameterInt64(const std::string &name, Int64 index);
  virtual UInt64 getParameterUInt64(const std::string &name, Int64 index);
  virtual Real32 getParameterReal32(const std::string &name, Int64 index);
  virtual Real64 getParameterReal64(const std::string &name, Int64 index);
  virtual bool getParameterBool(const std::string &name, Int64 index);

  virtual void setParameterInt32(const std::string &name, Int64 index,
                                 Int32 value);
  virtual void setParameterUInt32(const std::string &name, Int64 index,
                                  UInt32 value);
  virtual void setParameterInt64(const std::string &name, Int64 index,
                                 Int64 value);
  virtual void setParameterUInt64(const std::string &name, Int64 index,
                                  UInt64 value);
  virtual void setParameterReal32(const std::string &name, Int64 index,
                                  Real32 value);
  virtual void setParameterReal64(const std::string &name, Int64 index,
                                  Real64 value);
  virtual void setParameterBool(const std::string &name, Int64 index,
                                bool value);

  virtual void getParameterArray(const std::string &name, Int64 index,
                                 Array &array);
  virtual void setParameterArray(const std::string &name, Int64 index,
                                 const Array &array);

  virtual void setParameterString(const std::string &name, Int64 index,
                                  const std::string &s);
  virtual std::string getParameterString(const std::string &name, Int64 index);

  /* -------- Methods that must be implemented by subclasses -------- */

  /**
   * Region implimentations must implement createSpec().
   * Can't declare a static method in an interface. But RegionFactory
   * expects to find this method. Caller gets ownership of Spec pointer.
   * The returned spec pointer is cached by RegionImplFactory in regionSpecMap
   * which is a map of shared_ptr's.
   */
  // static Spec* createSpec();

  // overridden by including the macro CerealAdapter in subclass.
  virtual void cereal_adapter_save(ArWrapper& a) const {};
  virtual void cereal_adapter_load(ArWrapper& a) {};

  // NOTE: all internal regions must implement an override of operator== by convention for unit testing.  
  //            Customer written regions do not require it.
  virtual bool operator==(const RegionImpl &other) const { 
    NTA_THROW << "operator== not implmented for region "+getName(); 
  };
  virtual inline bool operator!=(const RegionImpl &other) const {
    return !operator==(other);
  }


  /**
    * Inputs/Outputs are made available in initialize()
    * It is always called after the constructor (or load from serialized state)
    */
  virtual void initialize() = 0;

  // Compute outputs from inputs and internal state
  virtual void compute() = 0;

  /* -------- Methods that may be overridden by subclasses -------- */

  // Execute a command
  virtual std::string executeCommand(const std::vector<std::string> &args,
                                     Int64 index);


  // Buffer size (in elements) of the given input/output.
  // It is the total element count.
  // This method is called only for buffers whose size is not
  // specified in the Spec.  This is used to allocate
  // buffers during initialization.  New implementations should instead
  // override askImplForOutputDimensions() or askImplForInputDimensions()
  // and return a full dimension.
  // Return 0 for outputs that are not used or size does not matter.
  virtual size_t getNodeInputElementCount(const std::string &outputName) const {
    return Dimensions::DONTCARE;
  }
  virtual size_t getNodeOutputElementCount(const std::string &outputName) const {
    return Dimensions::DONTCARE;
  }


  // The dimensions for the specified input or output.  This is called by
  // Link when it allocates buffers during initialization.
  // If this region sets topology (an SP for example) and will be
  // setting the dimensions (i.e. from parameters) then
  // return the dimensions that should be placed in its Array buffer.
  // Return an isDontCare() Dimension if this region should inherit
  // dimensions from elsewhere.
  //
  // If this is not overridden, the default implementation will call
  // getNodeOutputElementCount() or getNodeInputElementCount() to obtain
  // a 1D dimension for this input/output.
  virtual Dimensions askImplForInputDimensions(const std::string &name);
  virtual Dimensions askImplForOutputDimensions(const std::string &name);


  /**
   * Array-valued parameters may have a size determined at runtime.
   * This method returns the number of elements in the named parameter.
   * If parameter is not an array type, may throw an exception or return 1.
   *
   * Must be implemented only if the node has one or more array
   * parameters with a dynamically-determined length.
   */
  virtual size_t getParameterArrayCount(const std::string &name, Int64 index);

  /**
   * Set Global dimensions on a region.
   * Normally a Region Impl will use this to set the dimensions on the default output.
   * This cannot be used to override a fixed buffer setting in the Spec.
   * Args: dim   - The dimensions to set
   */
  virtual void setDimensions(Dimensions dim) { dim_ = std::move(dim); }
  virtual Dimensions getDimensions() const { return dim_; }

  virtual ValueMap ValidateParameters(const ValueMap &vm, Spec* ns);

  static Spec *parseSpec(const std::string &yaml);

protected:
  // A pointer to the Region object. This is the portion visible
	// to the applications.  This class and it's subclasses are the
	// hidden implementations behind the Region class.
	// Note: this cannot be a shared_ptr. Its pointer is passed via
	//       the API so it must be a bare pointer so we don't have
	//       a copy of the shared_ptr held by the Collection in Network.
	//       This pointer must NOT be deleted.
  Region* region_;

  // A local copy of the spec.
  std::shared_ptr<Spec> spec_;

  // Region level dimensions.  This is set by the parameter "{dim: [2,3]}"
  // or by region->setDimensions(d);
  // A region implementation may use this for whatever it wants but it is normally
  // applied to the default output buffer.
  Dimensions dim_;


  // These methods provide access to inputs and outputs
  // They raise an exception if the named input or output is
  // not found.
  inline bool hasOutput(const std::string &name) const { return region_->hasOutput(name); }
  inline bool hasInput(const std::string &name) const { return region_->hasInput(name); }

  std::shared_ptr<Input> getInput(const std::string &name) const;
  std::shared_ptr<Output> getOutput(const std::string &name) const;
  Dimensions getInputDimensions(const std::string &name="") const;
  Dimensions getOutputDimensions(const std::string &name="") const;

};

} // namespace htm

#endif // NTA_REGION_IMPL_HPP
