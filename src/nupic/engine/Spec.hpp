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
Definition of Spec data structures
*/

#ifndef NTA_SPEC_HPP
#define NTA_SPEC_HPP

#include <map>
#include <nupic/ntypes/Collection.hpp>
#include <nupic/types/Types.hpp>
#include <string>

namespace nupic {
class InputSpec {
public:
  InputSpec() {}
  InputSpec(std::string description, NTA_BasicType dataType, UInt32 count,
            bool required, bool regionLevel, bool isDefaultInput,
            bool requireSplitterMap = true, bool sparse = false);
  bool operator==(const InputSpec &other) const;
  inline bool operator!=(const InputSpec &other) const {
    return !operator==(other);
  }
  std::string description;
  NTA_BasicType dataType;
  // TBD: Omit? isn't it always of unknown size?
  // 1 = scalar; > 1 = array of fixed sized; 0 = array of unknown size
  UInt32 count;
  // TBD. Omit? what is "required"? Is it ok to be zero length?
  bool required;
  bool regionLevel;
  bool isDefaultInput;
  bool requireSplitterMap;
  bool sparse;
};

class OutputSpec {
public:
  OutputSpec() {}
  OutputSpec(std::string description, const NTA_BasicType dataType,
             size_t count, bool regionLevel, bool isDefaultOutput,
             bool sparse = false);
  bool operator==(const OutputSpec &other) const;
  inline bool operator!=(const OutputSpec &other) const {
    return !operator==(other);
  }
  std::string description;
  NTA_BasicType dataType;
  // Size, in number of elements. If size is fixed, specify it here.
  // Value of 0 means it is determined dynamically
  size_t count;
  bool regionLevel;
  bool isDefaultOutput;
  bool sparse;
};

class CommandSpec {
public:
  CommandSpec() {}
  CommandSpec(std::string description);
  bool operator==(const CommandSpec &other) const;
  inline bool operator!=(const CommandSpec &other) const {
    return !operator==(other);
  }
  std::string description;
};

class ParameterSpec {
public:
  typedef enum { CreateAccess, ReadOnlyAccess, ReadWriteAccess } AccessMode;

  ParameterSpec() {}
  /**
   * @param defaultValue -- a JSON-encoded value
   */
  ParameterSpec(std::string description, NTA_BasicType dataType, size_t count,
                std::string constraints, std::string defaultValue,
                AccessMode accessMode);
  bool operator==(const ParameterSpec &other) const;
  inline bool operator!=(const ParameterSpec &other) const {
    return !operator==(other);
  }
  std::string description;

  // [open: current basic types are bytes/{u}int16/32/64, real32/64, BytePtr. Is
  // this the right list? Should we have std::string, jsonstd::string?]
  NTA_BasicType dataType;
  // 1 = scalar; > 1 = array o fixed sized; 0 = array of unknown size
  // TODO: should be size_t? Serialization issues?
  size_t count;
  std::string constraints;
  std::string defaultValue; // JSON representation; empty std::string means
                            // parameter is required
  AccessMode accessMode;
};

struct Spec {
  // Return a printable string with Spec information
  // TODO: should this be in the base API or layered? In the API right
  // now since we do not build layered libraries.
  std::string toString() const;
  bool operator==(const Spec &other) const;
  inline bool operator!=(const Spec &other) const { return !operator==(other); }
  // Some RegionImpls support only a single node in a region.
  // Such regions always have dimension [1]
  bool singleNodeOnly;

  // Description of the node as a whole
  std::string description;

  Collection<InputSpec> inputs;
  Collection<OutputSpec> outputs;
  Collection<CommandSpec> commands;
  Collection<ParameterSpec> parameters;

#ifdef NTA_INTERNAL

  Spec();

  // TODO: decide whether/how to wrap these
  std::string getDefaultOutputName() const;
  std::string getDefaultInputName() const;

  // TODO: need Spec validation, to make sure
  // that default input/output are defined
  // Currently this is checked in getDefault*, above

#endif // NTA_INTERNAL
};

} // namespace nupic

#endif // NTA_SPEC_HPP
