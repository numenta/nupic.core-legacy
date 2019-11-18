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
Definition of Spec data structures
*/

#ifndef NTA_SPEC_HPP
#define NTA_SPEC_HPP

#include <map>
#include <htm/ntypes/Collection.hpp>
#include <htm/types/Types.hpp>
#include <string>

namespace htm {


class InputSpec {
public:
  InputSpec(std::string description = "", 
            NTA_BasicType dataType = NTA_BasicType_SDR,
            UInt32 count = 0u,
            bool required = false,
            bool regionLevel = true,
            bool isDefaultInput = false);
    bool operator==(const InputSpec &other) const;
    inline bool operator!=(const InputSpec &other) const {
    return !operator==(other);
  }
  std::string description;   // description of input
	
  NTA_BasicType dataType; // declare type of input

  // width of buffer if fixed. 0 means variable.
  // If non-zero positive value it means this region was developed
	// to accept a fixed sized 1D array only.
  UInt32 count;             
	
  bool required;             // true if input must be connected.
	
  bool regionLevel;          // if true, this means this input can propagate its 
                             // dimensions to/from the region's dimensions.
	
  bool isDefaultInput;       // if True, assume this if input name not given 
	                           // in functions involving inputs of a region.
};

class OutputSpec {
public:
  OutputSpec(std::string description = "", 
             NTA_BasicType dataType = NTA_BasicType_SDR,
             size_t count = 0u,              // set size of buffer, 0 means unknown size.
             bool regionLevel = true,
             bool isDefaultOutput = false);
    bool operator==(const OutputSpec &other) const;
    inline bool operator!=(const OutputSpec &other) const {
    return !operator==(other);
  }
  std::string description;   // description of output
	
	NTA_BasicType dataType; // The type of the output buffer.

  size_t count;              // Size, in number of elements. If size is fixed.  
	                           // If non-zero value it means this region 
														 // was developed to output a fixed sized 1D array only.
                             // if 0, call askImplForOutputDimensions() to get dimensions.

  bool regionLevel;          // If true, this output is can get its dimensions from
                             // the region dimensions.

  bool isDefaultOutput;      // if true, use this output for region if output name not given
	                           // in functions involving outputs on a region.
};

class CommandSpec {
public:
  CommandSpec(std::string description = "");
  bool operator==(const CommandSpec &other) const;
  inline bool operator!=(const CommandSpec &other) const {
    return !operator==(other);
  }
  std::string description;
};

class ParameterSpec {
public:
  typedef enum { CreateAccess, ReadOnlyAccess, ReadWriteAccess } AccessMode;

  /**
   * @param defaultValue -- a JSON-encoded value
   */
  ParameterSpec(std::string description = "", 
                NTA_BasicType dataType = NTA_BasicType_Real64, 
                size_t count = 1u,
                std::string constraints = "", 
                std::string defaultValue = "0", 
                AccessMode accessMode = AccessMode::CreateAccess);
  bool operator==(const ParameterSpec &other) const;
  inline bool operator!=(const ParameterSpec &other) const {
    return !operator==(other);
  }
  std::string description;

  // current basic types are string, Byte, {U}Int16/32/64, Real32/64, Bool. 
  NTA_BasicType dataType;
  // 1 = scalar; > 1 = array o fixed sized; 0 = array of unknown size
  // TODO: should be size_t? Serialization issues?
  size_t count;
  std::string constraints;
  std::string defaultValue; // JSON representation; empty std::string means
                            // parameter is required
  AccessMode accessMode;
};

class Spec {
public:
  // Return a printable string with Spec information
  // TODO: should this be in the base API or layered? In the API right
  // now since we do not build layered libraries.
  std::string toString() const;
  bool operator==(const Spec &other) const;
  inline bool operator!=(const Spec &other) const { return !operator==(other); }
  // Some RegionImpls support only a single node in a region.
  // Such regions always have dimension [1]
  bool singleNodeOnly;

  // Region type name
  std::string name;

  // Description of the node as a whole
  std::string description;

  // containers for the components of the spec.
  Collection<InputSpec> inputs;
  Collection<OutputSpec> outputs;
  Collection<CommandSpec> commands;
  Collection<ParameterSpec> parameters;


  Spec();

  std::string getDefaultOutputName() const;
  std::string getDefaultInputName() const;

  // a value that applys to the count field in inputs, outputs, parameters.
  // It means that the field is an array and its size is not fixed.
  static const size_t VARIABLE = 0u;

  // a value that applys to the count field in inputs, outputs, parameters.
  // It means that the field not an array and has a single scaler value.
  static const size_t SCALER = 1u;


  // An alternative format.  Parses a Yaml or JSON format string
  // into a Spec structure.
  void parseSpec(const std::string &yaml);


};

std::ostream& operator<< (std::ostream& stream, const Spec& self);

} // namespace htm

#endif // NTA_SPEC_HPP
