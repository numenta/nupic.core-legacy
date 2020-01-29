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

#include <iostream>

#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/engine/RegionImpl.hpp>

namespace htm {

RegionImpl::RegionImpl(Region *region) : region_(region) {}

RegionImpl::~RegionImpl() {}

// convenience method
std::string RegionImpl::getType() const { return region_->getType(); }

std::string RegionImpl::getName() const { return region_->getName(); }


/* ------------- Parameter support --------------- */
// By default, all typed getParameter calls forward to the
// untyped getParameter that serializes to a buffer

// Use macros to implement these methods.
// This is similar to a template + explicit instantiation, but
// templated methods can't be virtual and thus can't be
// overridden by subclasses.

#define getParameterInternalT(MethodT, Type)                                   \
  Type RegionImpl::getParameter##MethodT(const std::string &name,              \
                                         Int64 index) {                        \
    if (!region_->getSpec()->parameters.contains(name))                        \
      NTA_THROW << "getParameter" #Type ": Region type " << getType()          \
	            << ", parameter " << name                                      \
                << " does not exist in region spec";                              \
    ParameterSpec p = region_->getSpec()->parameters.getByName(name);          \
    if (p.dataType != NTA_BasicType_##MethodT)                                 \
      NTA_THROW << "getParameter" #Type ": Region type " << getType()          \
	            << ", parameter " << name                                      \
                << " is of type " << BasicType::getName(p.dataType)            \
                << " not " #Type;                                              \
    NTA_THROW << "getParameter" #Type " --  parameter '"       \
                << name << "' on region of type " << getType() << " not implemented"; \
  }

#define getParameterT(Type) getParameterInternalT(Type, Type)

getParameterT(Int32);
getParameterT(UInt32);
getParameterT(Int64);
getParameterT(UInt64) getParameterT(Real32);
getParameterT(Real64);
getParameterInternalT(Bool, bool);

#define setParameterInternalT(MethodT, Type)                                   \
  void RegionImpl::setParameter##MethodT(const std::string &name, Int64 index, \
                                         Type value) {                         \
    if (!region_->getSpec()->parameters.contains(name))                        \
      NTA_THROW << "setParameter" #Type ": Region type " << getType()          \
	            << ", parameter " << name                                      \
                << " does not exist in Spec";                                  \
    ParameterSpec p = region_->getSpec()->parameters.getByName(name);          \
    if (p.dataType != NTA_BasicType_##MethodT)                                 \
      NTA_THROW << "setParameter" #Type ": Region type " << getType()          \
	            << ", parameter " << name                                      \
                << " is of type " << BasicType::getName(p.dataType)            \
                << " not " #Type;                                              \
    if (p.accessMode != ParameterSpec::ReadWriteAccess)                        \
      NTA_THROW << "setParameter" #Type " --  parameter '" << name << " is Readonly"; \
    NTA_THROW << "setParameter" #Type " -- Region type " << getType()          \
			    << ", parameter '" << name << "' not implemented";             \
  }

#define setParameterT(Type) setParameterInternalT(Type, Type)

setParameterT(Int32);
setParameterT(UInt32);
setParameterT(Int64);
setParameterT(UInt64);
setParameterT(Real32);
setParameterT(Real64);
setParameterInternalT(Bool, bool);

void RegionImpl::getParameterArray(const std::string &name, Int64 index, Array &array) {
  if (!region_->getSpec()->parameters.contains(name))
      NTA_THROW << "setParameterArray: parameter " << name
                << " does not exist in Spec";
  ParameterSpec p = region_->getSpec()->parameters.getByName(name);

  NTA_THROW << "getParameterArray: parameter '" << name
            << "' an array with a type of "
			<< BasicType::getName(p.dataType)
			<< " is found in the region spec but is not implemented.";
}

void RegionImpl::setParameterArray(const std::string &name, Int64 index, const Array &array) {
  if (!region_->getSpec()->parameters.contains(name))
      NTA_THROW << "setParameterArray: parameter " << name
                << " does not exist in Spec for this region.";
  ParameterSpec p = region_->getSpec()->parameters.getByName(name);
  if (p.dataType != array.getType()) {
      NTA_THROW << "setParameterArray: parameter " << name
                << " is of type " << BasicType::getName(p.dataType)
                << " not " << BasicType::getName(array.getType());
  }
  NTA_THROW	<< "setParameterArray: parameter '" << name
            << " is found in the spec for " << getType() <<" but is not implemented.";
}

void RegionImpl::setParameterString(const std::string &name, Int64 index,
                                    const std::string &s) {
  if (!region_->getSpec()->parameters.contains(name))
      NTA_THROW << "setParameterString: parameter " << name
                << " does not exist in Spec";
  ParameterSpec p = region_->getSpec()->parameters.getByName(name);
  if (p.dataType != NTA_BasicType_Byte) {
      NTA_THROW << "setParameterString: parameter " << name
                << " is of type " << BasicType::getName(p.dataType)
                << " not Byte (string)";
  }
  NTA_THROW << "setParameterString: parameter '" << name
			<< " is found in the spec for " << getType() <<" but is not implemented.";
}

std::string RegionImpl::getParameterString(const std::string &name,
                                           Int64 index) {
  if (!region_->getSpec()->parameters.contains(name))
      NTA_THROW << "getParameterString: parameter " << name
                << " does not exist in Spec";
  ParameterSpec p = region_->getSpec()->parameters.getByName(name);
  if (p.dataType != NTA_BasicType_Byte) {
      NTA_THROW << "getParameterString: parameter " << name
                << " is of type " << BasicType::getName(p.dataType)
                << " not Byte (string)";
  }
  NTA_THROW << "getParameterString: parameter '" << name
			<< " is found in the spec for " << getType() <<" but is not implemented.";
  return "";
}



size_t RegionImpl::getParameterArrayCount(const std::string &name,
                                          Int64 index) {
  // Default implementation for RegionImpls with no array parameters
  // that have a dynamic length.
  // std::map<std::string, ParameterSpec*>::iterator i =
  // nodespec_->parameters.find(name); if (i == nodespec_->parameters.end())

  if (!region_->getSpec()->parameters.contains(name)) {
    NTA_THROW << "getParameterArrayCount -- no parameter named '" << name
              << "' in node of type " << getType();
  }
  UInt32 count = (UInt32)region_->getSpec()->parameters.getByName(name).count;
  if (count == 0) {
    NTA_THROW << "Internal Error -- unknown element count for "
              << "node type " << getType() << ". The RegionImpl "
              << "implementation should override this method.";
  }

  return count;
}

Dimensions RegionImpl::askImplForInputDimensions(const std::string &name) {
  // Default implementation for Region Impl that did not override this.
  // This should return the input dimensions for this input, or a Dimension
  // of size 1, value 0 which means don't care.
  //
  // Since the region impl did not override this, we generate the Dimensions
  // based on the Spec.
  if (!region_->getSpec()->inputs.contains(name)) {
    NTA_THROW << "askImplForInputDimensions -- no input named '" << name
              << "' in region of type " << getType();
  }
  UInt32 count = (UInt32)region_->getSpec()->inputs.getByName(name).count;
  if (count == Spec::VARIABLE)
    count = (UInt32)getNodeInputElementCount(name);
  Dimensions dim;
  dim.push_back(count);   // if count == 0, it means don't care.
  return dim;
}

Dimensions RegionImpl::askImplForOutputDimensions(const std::string &name) {
  // Default implementation for Region Impl that did not override this.
  // This should return the input dimensions for this input, or a Dimension
  // of size 1, value 0 which means don't care.
  //
  // Since the region impl did not override this, we generate the Dimensions
  // based on the Spec.
  if (!region_->getSpec()->outputs.contains(name)) {
    NTA_THROW << "askImplForOutputDimensions -- no output named '" << name
              << "' in region of type " << getType();
  }
  auto ns = region_->getSpec()->outputs.getByName(name);

  UInt32 count = (UInt32)ns.count;
  if (count == Spec::VARIABLE) {
    // This is not a fixed size output.
    // ask the impl for a 1D size.
    count = (UInt32)getNodeOutputElementCount(name);

    // If this is a regionLevel output, use the region dimensions
    // provided that count is 0 or that count is same element size as regionLevel.
    if (ns.regionLevel && dim_.isSpecified()) {
      if (count == 0 || count == dim_.getCount())
      return dim_;
    }
  }
  Dimensions dim;
  dim.push_back(count);   // if count happens to be 0, it means don't care.
  return dim;
}


std::string RegionImpl::executeCommand(const std::vector<std::string> &args,Int64 index) {
  // This Region does not execute any Commands.
  return "";
}

// Provide data access for subclasses

std::shared_ptr<Input> RegionImpl::getInput(const std::string &name) const {
  return region_->getInput(name);
}

std::shared_ptr<Output> RegionImpl::getOutput(const std::string &name) const {
  auto out = region_->getOutput(name);
  NTA_CHECK(out != nullptr) << "Requested output not found: " << name;
  return out;
}
Dimensions RegionImpl::getInputDimensions(const std::string &name) const {
  return region_->getInputDimensions(name);
}
Dimensions RegionImpl::getOutputDimensions(const std::string &name) const {
  return region_->getOutputDimensions(name);
}

/**
 * Checks the parameters in the ValueMap and gives an error if it
 * is not consistant with the Spec.  If a field in the Spec is not given
 * in the ValueMap, insert it with its default value.
 * Returns a modifiable deep copy of the ValueMap with defaults inserted.
 * For optional use by C++ implemented Regions.
 */
ValueMap RegionImpl::ValidateParameters(const ValueMap &vm, Spec* ns) {
  
  ValueMap new_vm = vm.copy();

  // Look for parameters that don't belong
  for (auto p: new_vm) {
    std::string key = p.first;
    Value v = p.second;
    if (key == "dim")
      continue;
    if (!ns->parameters.contains(key))
      NTA_THROW << "Parameter '" << key << "' is not expected for region '" << getName() << "'.";
  }
  

  // Look for missing parameters and apply their default value.
  for (auto p : ns->parameters) {
    std::string key = p.first;
    ParameterSpec &ps = p.second;
    if (!ps.defaultValue.empty()) {
      if (new_vm.getString(key, "").length() == 0) {
        // a missing or empty parameter.
        new_vm[key] = ps.defaultValue;
      }
    }
  }
  
  return new_vm;

}



} // namespace htm
