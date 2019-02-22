/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <iostream>

#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/types/BasicType.hpp>

namespace nupic {

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
  if (p.dataType != array.getType()) {
      NTA_THROW << "setParameterArray: parameter " << name
                << " is of type " << BasicType::getName(p.dataType)
                << " not " << BasicType::getName(array.getType());
  }
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

// Must be overridden by subclasses
bool RegionImpl::isParameterShared(const std::string &name) {
  NTA_THROW << "RegionImpl::isParameterShared was not overridden in node type "
            << getType();
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

// Provide data access for subclasses

Input *RegionImpl::getInput(const std::string &name) const {
  return region_->getInput(name);
}

Output *RegionImpl::getOutput(const std::string &name) const {
  return region_->getOutput(name);
}

const Dimensions &RegionImpl::getDimensions() {
  return region_->getDimensions();
}

} // namespace nupic
