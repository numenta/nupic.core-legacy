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
 * Interface for the Region class.
 *
 * A region is a set of one or more "identical" nodes, implemented by a
 * RegionImpl"plugin". A region contains nodes.
 */

#ifndef NTA_REGION_HPP
#define NTA_REGION_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

// We need the full definitions because these
// objects are returned by value.
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/os/Timer.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/types/Types.hpp>

namespace htm {

class RegionImpl;
class Output;
class Input;
class Array;
class Spec;
class BundleIO;
class Timer;
class Network;

/**
 * Represents a set of one or more "identical" nodes in a Network.
 *
 * @nosubgrouping
 *
 * ### Constructors
 *
 * @note Region constructors are not available in the public API.
 * Internally regions are created and owned by Network.
 *
 */
class Region : public Serializable {
public:
  /**
   * @name Region information
   *
   * @{
   */

  /**
   * Get the network containing this region.
   *
   * @returns The network containing this region
   */
  Network *getNetwork();

  /**
   * Get the name of the region.
   *
   * @returns The region's name
   */
  std::string getName() const { return name_; }

  /**
   * @}
   *
   * @name Element interface methods
   *
   * @todo What does "Element interface methods" mean here?
   *
   * @{
   *
   */

  /**
   * Get the type of the region.
   *
   * @returns The node type as a string
   */
  std::string getType() const { return type_; }

  /**
   * Get the spec of the region.
   *
   * @returns The spec that describes this region
   */
  const std::shared_ptr<Spec> &getSpec() const { return spec_; }

  /**
   * Get the Spec of a region type without an instance.
   *
   * @param nodeType
   *        A region type as a string
   *
   * @returns The Spec that describes this region type
   */
  static const std::shared_ptr<Spec> &
  getSpecFromType(const std::string &nodeType);

  /**
   * @}
   *
   * @name Parameter getters and setters
   *
   * @{
   *
   */

  /**
   * Get the parameter value as a specific type.
   *
   * @param name
   *        The name of the parameter
   *
   * @returns The value of the parameter
   */
  Int32 getParameterInt32(const std::string &name) const;
  UInt32 getParameterUInt32(const std::string &name) const;
  Int64 getParameterInt64(const std::string &name) const;
  UInt64 getParameterUInt64(const std::string &name) const;
  Real32 getParameterReal32(const std::string &name) const;
  Real64 getParameterReal64(const std::string &name) const;
  bool getParameterBool(const std::string &name) const;

  /**
   * Set the parameter value of a specific type.
   *
   * @param name
   *        The name of the parameter
   *
   * @param value
   *        The value of the parameter
   */
  void setParameterInt32(const std::string &name, Int32 value);
  void setParameterUInt32(const std::string &name, UInt32 value);
  void setParameterInt64(const std::string &name, Int64 value);
  void setParameterUInt64(const std::string &name, UInt64 value);
  void setParameterReal32(const std::string &name, Real32 value);
  void setParameterReal64(const std::string &name, Real64 value);
  void setParameterBool(const std::string &name, bool value);

  /**
   * Get the parameter as an @c Array value.
   *
   * @param name
   *        The name of the parameter.  The name must be
   *        declared in the Region Spec as a parameter.
   *
   * @param[out] array
   *        The array to return the value of the parameter
   *
   * @a array is a container to hold a buffer or SDR. The Region Implementation
   * will allocate the buffer and copy the value of the parameter into it.
   *
   * A typical use might be that the caller would supply an
   * unallocated Array
   *
   * @code{.cpp}
   *     {
   *       // no buffer allocated
   *       Array buffer();
   *       getParameterArray("foo", buffer);
   *       getParameterArray("foo", buffer);
   *     } // Array destructor called -- frees the buffer
   * @endcode
   *
   */
  void getParameterArray(const std::string &name, Array &array) const;

  /**
   * Set the parameter to an @c Array value.
   *
   * @param name
   *        The name of the parameter.  The name must be
   *        declared in the Region Spec as a parameter.
   *
   * @param array
   *        The value of the parameter
   *
   *
   * @note @a array must be initialized before calling setParameterArray().
   *             if it is populated by passing in a vector, the buffer is allocated and data copied while creating the Array.
   *             and the buffer is freed when the Array's destructor is called.
   *
   *             if it is populated by passing in a type, pointer and length, the pointer becomes the buffer
   *             and is not freed when the Array's destructor is called.
   *
   * @code{.cpp}
   *     {
   *       std::vector<Real32> v1 = {1.0f, 2.0f, 3.0f};
   *       Array a1(v);      // type and size set based on vector, buffer allocated and data copied.
  *       setParameterArray("foo", a1);
   *     } // Array destructor called -- frees the buffer
   *     {
  *        Int32 q[3] = {1, 2, 3};
   *       Array buffer2(NTA_BasicType_Int32, q, 3);   // pointer q becomes the buffer
   *       setParameterArray("foo", buffer2);
   *     }  // Array destructor called -- buffer is not freed.
   * @endcode
   *
   */
  void setParameterArray(const std::string &name, const Array &array);

	
  /**
   * Get the number of elements in the array parameter's value.
   *
   * @param name
   *        The name of the parameter
	 */
  size_t getParameterArrayCount(const std::string &name);


  /**
   * Set the parameter to a @c std::string value.
   *
   * @param name
   *        The name of the parameter
   *
   * @param s
   *        The value of the parameter
   *
   * Strings are handled internally as Byte Arrays, but this interface
   * is clumsy. setParameterString() and getParameterString() internally use
   * byte arrays but converts to/from strings.
   *
   * setParameterString() is implemented with one copy (from the string into
   * the node) but getParameterString() requires a second copy so that there
   * are temporarily three copies of the data in memory (in the node,
   * in an internal Array object, and in the string returned to the user)
   *
   */
  void setParameterString(const std::string &name, const std::string &s);

  /**
   * Get the parameter as a @c std::string value.
   *
   * @param name
   *        The name of the parameter
   *
   * @returns
   *         The value of the parameter
   *
   * @see setParameterString()
   */
  std::string getParameterString(const std::string &name);

  /**
   * Tells whether the parameter is shared.
   *
   * @param name
   *        The name of the parameter
   *
   * @returns
   *        Whether the parameter exists
   *
   */
  bool isParameter(const std::string &name) const;

  /**
   * @}
   *
   * @name Inputs and outputs
   *
   * @{
   *
   */

  /**
   * Copies data into the inputs of this region, using
   * the links that are attached to each input.
   */
  void prepareInputs();

  /**
   * Get the input data.
   *
   *
   * @param inputName
   *        The name of the target input
   *
   * @returns An @c const Array that references the Input object's buffer.
   *        This buffer is shared with the Input object so when it changes the
   *        returned Array's buffer also changes.
   *        Note that this is read-only.
   */
  virtual const Array &getInputData(const std::string &inputName) const;

  /**
   * Get the output data.
   *
   * @param outputName
   *        The name of the target output
   *
   * @returns
   *        An @c const Array that references the output data buffer.
   *        This buffer is shared with the Output object so when it changes the
   *        returned Array's buffer also changes.
   *        Note that this is read-only.
   *        To obtain a writeable Array use
   *			  region->getOutput(name)->getData();
   */
  virtual const Array &getOutputData(const std::string &outputName) const;

  /**
   * @}
   *
   * @name Operations
   *
   * @{
   *
   */

  /**
   * Request the underlying region to execute a command.
   *
   * @param args
   *        A list of strings that the actual region will interpret.
   *        The first string is the command name. The other arguments are
   * optional.
   *
   * @returns
   *        The result value of command execution is a string determined
   *          by the underlying region.
   */
  virtual std::string executeCommand(const std::vector<std::string> &args);

  /**
   * Perform one step of the region computation.
   */
  void compute();

  /**
   * @}
   *
   * @name Profiling
   *
   * @{
   *
   */

  /**
   * Enable profiling of the compute and execute operations
   */
  void enableProfiling();

  /**
   * Disable profiling of the compute and execute operations
   */
  void disableProfiling();

  /**
   * Reset the compute and execute timers
   */
  void resetProfiling();

  /**
   * Get the timer used to profile the compute operation.
   *
   * @returns
   *        The Timer object used to profile the compute operation
   */
  const Timer &getComputeTimer() const;

  /**
   * Get the timer used to profile the execute operation.
   *
   * @returns
   *        The Timer object used to profile the execute operation
   */
  const Timer &getExecuteTimer() const;

  bool operator==(const Region &other) const;
  inline bool operator!=(const Region &other) const {
    return !operator==(other);
  }

  /**
   * @}
   */

  // Internal methods.

  // New region from parameter spec
  Region(std::string name, const std::string &type,
         const std::string &nodeParams, Network *network = nullptr);

  Region(Network *network); // An empty region for deserialization.
  Region(); // A default constructor for region for deserialization.

  virtual ~Region();

  void initialize();

  bool isInitialized() const { return initialized_; }

  // Used by RegionImpl to get inputs/outputs
  Output *getOutput(const std::string &name) const;

  Input *getInput(const std::string &name) const;

  const std::map<std::string, Input *> &getInputs() const;

  const std::map<std::string, Output *> &getOutputs() const;

  void clearInputs();

  // The following methods are called by Network in initialization

  // Configure and initialize links.
  void evaluateLinks();
  size_t getNodeOutputElementCount(const std::string &name);
  size_t getNodeInputElementCount(const std::string &name);
  Dimensions askImplForOutputDimensions(const std::string &name) const;
  Dimensions askImplForInputDimensions(const std::string &name) const;
  Dimensions getInputDimensions(std::string name="") const;
  Dimensions getOutputDimensions(std::string name="") const;
  void setInputDimensions(std::string name, const Dimensions& dim);
  void setOutputDimensions(std::string name, const Dimensions& dim);

  /**
   * Set Global dimensions on a region.
   * Normally a Region Impl will use this to set the dimensions on the default output.
   * This cannot be used to override a fixed buffer setting in the Spec.
   * Args: dim   - The dimensions to set
   */
  void setDimensions(Dimensions dim);
  Dimensions getDimensions() const;


  bool hasOutgoingLinks() const;

  // These methods are needed for teardown choreography
  // in Network::~Network()
  // It is an error to call any region methods after uninitialize()
  // except removeAllIncomingLinks and ~Region
  void uninitialize();

  void removeAllIncomingLinks();

  // TODO: sort our phases api. Users should never call Region::setPhases
  // and it is here for serialization only.
  void setPhases(std::set<UInt32> &phases);

  std::set<UInt32> &getPhases();


  // These must be implemented for serialization.
  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    ar(cereal::make_nvp("name", name_),
       cereal::make_nvp("nodeType", type_),
       cereal::make_nvp("initialized", initialized_),
       cereal::make_nvp("phases", phases_));
    ar(cereal::make_nvp("dim", getDimensions()));

    std::map<std::string, Dimensions> outDims;
    std::map<std::string, Dimensions> inDims;
    getDims_(outDims, inDims);
    ar(cereal::make_nvp("output_dims", outDims));
    ar(cereal::make_nvp("input_dims",  inDims));
		
		// save the output buffers
    std::map<std::string, Array> buffers;
    getOutputBuffers_(buffers);
	  ar(cereal::make_nvp("outputs", buffers));

    // Now serialize the RegionImpl plugin.
    ArWrapper arw(&ar);
    serializeImpl(arw);
  }


  // FOR Cereal Deserialization
  // Note: custom region implementations must be registered
  //       before deserializing a region.
  template<class Archive>
  void load_ar(Archive& ar) {
    Dimensions dim;
    bool init;
    ar(cereal::make_nvp("name", name_));
    ar(cereal::make_nvp("nodeType", type_));
    ar(cereal::make_nvp("initialized", init));
    ar(cereal::make_nvp("phases", phases_));
    ar(cereal::make_nvp("dim", dim));

    std::map<std::string, Dimensions> outDims;
    std::map<std::string, Dimensions> inDims;
    ar(cereal::make_nvp("output_dims", outDims));
    ar(cereal::make_nvp("input_dims",  inDims));

    // deserialize the output buffers for this region.
    std::map<std::string, Array> buffers;
    ar(cereal::make_nvp("outputs", buffers));
    restoreOutputBuffers_(buffers);
    loadDims_(outDims, inDims);

    // deserialize the RegionImpl plugin and its algorithm
    ArWrapper arw(&ar);
    deserializeImpl(arw);

    // set the region dimensions.
    initialized_ = false; // setDimensions requires initialization off.
    setDimensions(dim);
    initialized_ = init;
  }

  friend class Network;  // so Network can set Network* network_; during addRegion( ).
  friend std::ostream &operator<<(std::ostream &f, const Region &r);


private:
  //Region(Region &){}  // copy not allowed

  // local functions
  void createInputsAndOutputs_();
  void getOutputBuffers_(std::map<std::string, Array>& buffers) const;
  void restoreOutputBuffers_(const std::map<std::string, Array>& buffers);
  void getDims_(std::map<std::string,Dimensions>& outDims,
               std::map<std::string,Dimensions>& inDims) const;
  void loadDims_(std::map<std::string,Dimensions>& outDims,
               std::map<std::string,Dimensions>& inDims) const;
  void serializeImpl(ArWrapper& ar) const;
  void deserializeImpl(ArWrapper& ar);

  std::string name_;

  // pointer to the "plugin"; owned by Region
  std::shared_ptr<RegionImpl> impl_;
  std::string type_;
  std::shared_ptr<Spec> spec_;

  typedef std::map<std::string, Output *> OutputMap;
  typedef std::map<std::string, Input *> InputMap;

  OutputMap outputs_;
  InputMap inputs_;
  // used for serialization only
  std::set<UInt32> phases_;
  bool initialized_;

  // Region contains a backpointer to network_ only to be able
  // to retrieve the containing network via getNetwork() for inspectors.
  // The implementation should not use network_ in any other methods.
  // This cannot be a shared_ptr.
  Network *network_;

  // Profiling related methods and variables.
  bool profilingEnabled_;
  Timer computeTimer_;
  Timer executeTimer_;
};

} // namespace htm

#endif // NTA_REGION_HPP
