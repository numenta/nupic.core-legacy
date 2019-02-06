/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
Implementation of the Region class

Methods related to parameters are in Region_parameters.cpp
Methods related to inputs and outputs are in Region_io.cpp

*/

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include <nupic/engine/Input.hpp>
#include <nupic/engine/Link.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/BasicType.hpp>


namespace nupic {

class GenericRegisteredRegionImpl;

// Create region from parameter spec
Region::Region(std::string name, const std::string &nodeType,
               const std::string &nodeParams, Network *network)
    : name_(std::move(name)), type_(nodeType), initialized_(false),
      enabledNodes_(nullptr), network_(network), profilingEnabled_(false) {
  // Set region info before creating the RegionImpl so that the
  // Impl has access to the region info in its constructor.
  RegionImplFactory &factory = RegionImplFactory::getInstance();
  spec_ = factory.getSpec(nodeType);


  // Dimensions start off as unspecified, but if
  // the RegionImpl only supports a single node, we
  // can immediately set the dimensions.
  if (spec_->singleNodeOnly)
    dims_.push_back(1);
  // else dims_ = []

  impl_.reset(factory.createRegionImpl(nodeType, nodeParams, this));
}

Region::Region(Network *net) {
      network_ = net;
      impl_ = nullptr;
      initialized_ = false;
      profilingEnabled_ = false;
	  enabledNodes_ = nullptr;
    } // for deserialization of region.


Network *Region::getNetwork() { return network_; }

void Region::createInputsAndOutputs_() {
  // Note: had to pass in a shared_ptr to itself so we can pass it to Inputs & Outputs.
  // Create all the outputs for this node type. By default outputs are zero size
  for (size_t i = 0; i < spec_->outputs.getCount(); ++i) {
    const std::pair<std::string, OutputSpec> &p = spec_->outputs.getByIndex(i);
    std::string outputName = p.first;
    const OutputSpec &os = p.second;
    auto output = new Output(this, os.dataType, os.regionLevel);
    outputs_[outputName] = output;
    // keep track of name in the output also -- see note in Region.hpp
    output->setName(outputName);
  }

  // Create all the inputs for this node type.
  for (size_t i = 0; i < spec_->inputs.getCount(); ++i) {
    const std::pair<std::string, InputSpec> &p = spec_->inputs.getByIndex(i);
    std::string inputName = p.first;
    const InputSpec &is = p.second;

    auto input = new Input(this, is.dataType, is.regionLevel);
    inputs_[inputName] = input;
    // keep track of name in the input also -- see note in Region.hpp
    input->setName(inputName);
  }
}

bool Region::hasOutgoingLinks() const {
  for (const auto &elem : outputs_) {
    if (elem.second->hasOutgoingLinks()) {
      return true;
    }
  }
  return false;
}

Region::~Region() {
  if (initialized_)
    uninitialize();

  // If there are any links connected to our outputs, this should fail.
  // We catch this error in the Network class and give the
  // user a good error message (regions may be removed either in
  // Network::removeRegion or Network::~Network())
  for (auto &elem : outputs_) {
    delete elem.second;
    elem.second = nullptr;
  }
  outputs_.clear();

  clearInputs(); // just in case there are some still around.

  // Note: the impl will be deleted when the region goes out of scope.
}

void Region::clearInputs() {
  for (auto &input : inputs_) {
    auto &links = input.second->getLinks();
    for (auto &link : links) {
      	link->getSrc().removeLink(link); // remove it from the Output object.
    }
	links.clear();
    delete input.second; // This is an Input object. Its destructor deletes the links.
    input.second = nullptr;
  }
  inputs_.clear();
}

void Region::initialize() {

  if (initialized_)
    return;

  impl_->initialize();
  initialized_ = true;
}


const std::shared_ptr<Spec>& Region::getSpecFromType(const std::string &nodeType) {
  RegionImplFactory &factory = RegionImplFactory::getInstance();
  return factory.getSpec(nodeType);
}


const Dimensions &Region::getDimensions() const { return dims_; }

void Region::enable() {
  NTA_THROW << "Region::enable not implemented (region name: " << getName()
            << ")";
}

void Region::disable() {
  NTA_THROW << "Region::disable not implemented (region name: " << getName()
            << ")";
}

std::string Region::executeCommand(const std::vector<std::string> &args) {
  std::string retVal;
  if (args.size() < 1) {
    NTA_THROW << "Invalid empty command specified";
  }

  if (profilingEnabled_)
    executeTimer_.start();

  retVal = impl_->executeCommand(args, (UInt64)(-1));

  if (profilingEnabled_)
    executeTimer_.stop();

  return retVal;
}

void Region::compute() {
  if (!initialized_)
    NTA_THROW << "Region " << getName()
              << " unable to compute because not initialized";

  if (profilingEnabled_)
    computeTimer_.start();

  impl_->compute();

  if (profilingEnabled_)
    computeTimer_.stop();

  return;
}

/**
 * These internal methods are called by Network as
 * part of initialization.
 */

size_t Region::evaluateLinks() {
  size_t nIncompleteLinks = 0;
  for (auto &elem : inputs_) {
    nIncompleteLinks += (elem.second)->evaluateLinks();
  }
  return nIncompleteLinks;
}

std::string Region::getLinkErrors() const {

  std::stringstream ss;
  for (const auto &elem : inputs_) {
    const std::vector<std::shared_ptr<Link>> &links = elem.second->getLinks();
    for (const auto &link : links) {
      if ((link)->getSrcDimensions().isUnspecified() ||
          (link)->getDestDimensions().isUnspecified()) {
        ss << (link)->toString() << "\n";
      }
    }
  }

  return ss.str();
}

size_t Region::getNodeOutputElementCount(const std::string &name) {
  // Use output count if specified in nodespec, otherwise
  // ask the Impl
  NTA_CHECK(spec_->outputs.contains(name)) << "No output named '" << name << "' in the spec";
  size_t count = spec_->outputs.getByName(name).count;
  if (count == 0) {
    try {
      count = impl_->getNodeOutputElementCount(name);
    } catch (Exception &e) {
      NTA_THROW << "Internal error -- the size for the output " << name
                << "is unknown. : " << e.what();
    }
  }

  return count;
}

void Region::initOutputs() {
  // Called by Network during initialization.
  // Some outputs are optional. These outputs will have 0 elementCount in the
  // node spec and also return 0 from impl->getNodeOutputElementCount(). These
  // outputs still appear in the output map, but with an array size of 0. All
  // other outputs we initialize to size determined by spec or by impl.

  for (auto &elem : outputs_) {
    const std::string &name = elem.first;

    size_t count = 0;
    try {
      count = getNodeOutputElementCount(name);
    } catch (nupic::Exception &e) {
      NTA_THROW << "Internal error -- unable to get size of output " << name
                << " : " << e.what();
    }
    elem.second->initialize(count);
  }
}

void Region::initInputs() const {
  auto i = inputs_.begin();
  for (; i != inputs_.end(); i++) {
    i->second->initialize();
  }
}

void Region::setDimensions(Dimensions &newDims) {
  // Can only set dimensions one time
  if (dims_ == newDims)
    return;

  if (dims_.isUnspecified()) {
    if (newDims.isDontcare()) {
      NTA_THROW << "Invalid attempt to set region dimensions to dontcare value";
    }

    if (!newDims.isValid()) {
      NTA_THROW << "Attempt to set region dimensions to invalid value:"
                << newDims.toString();
    }

    dims_ = newDims;
    dimensionInfo_ = "Specified explicitly in setDimensions()";
  } else {
    NTA_THROW << "Attempt to set dimensions of region " << getName() << " to "
              << newDims.toString() << " but region already has dimensions "
              << dims_.toString();
  }

}


void Region::setDimensionInfo(const std::string &info) {
  dimensionInfo_ = info;
}

const std::string &Region::getDimensionInfo() const { return dimensionInfo_; }

void Region::removeAllIncomingLinks() {
  InputMap::const_iterator i = inputs_.begin();
  for (; i != inputs_.end(); i++) {
    auto &links = i->second->getLinks();
    while (links.size() > 0) {
      i->second->removeLink(links[0]);
    }
  }
}

void Region::uninitialize() { initialized_ = false; }

void Region::setPhases(std::set<UInt32> &phases) { phases_ = phases; }

std::set<UInt32> &Region::getPhases() { return phases_; }


void Region::save(std::ostream &f) const {
  f << "{\n";
  f << "name: " << name_ << "\n";
  f << "nodeType: " << type_ << "\n";
  f << "dimensions: [ " << dims_.size() << "\n";
  for (UInt32 d : dims_) {
	  f << d << " ";
  }
  f << "]\n";
  f << "dimensionInfo: " << dimensionInfo_ << "\n";

  f << "phases: [ " << phases_.size() << "\n";
  for (const auto &phases_phase : phases_) {
      f << phases_phase << " ";
  }
  f << "]\n";
  f << "RegionImpl:\n";
  // Now serialize the RegionImpl plugin.
  BundleIO bundle(&f);
  impl_->serialize(bundle);

  f << "}\n";
}

void Region::load(std::istream &f) {
  char bigbuffer[5000];
  std::string tag;
  Size count;

  // Each region is a map -- extract the 5 values in the map
  f >> tag;
  NTA_CHECK(tag == "{") << "bad region entry (not a map)";

  // 1. name
  f >> tag;
  NTA_CHECK(tag == "name:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  name_ = bigbuffer;

  // 2. nodeType
  f >> tag;
  NTA_CHECK(tag == "nodeType:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  type_ = bigbuffer;

  // 3. dimensions
  f >> tag;
  NTA_CHECK(tag == "dimensions:");
  f >> tag;
  NTA_CHECK(tag == "[") << "Expecting a sequence.";
  f >> count;
  for (size_t i = 0; i < count; i++)
  {
    UInt32 val;
    f >> val;
    dims_.push_back(val);
  }
  f >> tag;
  NTA_CHECK(tag == "]") << "Expecting end of a sequence.";
  f >> tag;
  NTA_CHECK(tag == "dimensionInfo:") << "Expecting dimensionInfo";
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  dimensionInfo_ = bigbuffer;

  // 4. phases
  f >> tag;
  NTA_CHECK(tag == "phases:");
  f >> tag;
  NTA_CHECK(tag == "[") << "Expecting a sequence.";
  f >> count;
  phases_.clear();
  for (Size i = 0; i < count; i++)
  {
    UInt32 val;
    f >> val;
    phases_.insert(val);
  }
  f >> tag;
  NTA_CHECK(tag == "]") << "Expected end of sequence of phases.";

  // 5. impl
  f >> tag;
  NTA_CHECK(tag == "RegionImpl:") << "Expected beginning of RegionImpl.";
  f.ignore(1);

  RegionImplFactory &factory = RegionImplFactory::getInstance();
  spec_ = factory.getSpec(type_);
  createInputsAndOutputs_();

  BundleIO bundle(&f);
  impl_.reset(factory.deserializeRegionImpl(type_, bundle, this));

  f >> tag;
  NTA_CHECK(tag == "}") << "Expected end of region. Found '" << tag << "'.";
}



void Region::enableProfiling() { profilingEnabled_ = true; }

void Region::disableProfiling() { profilingEnabled_ = false; }

void Region::resetProfiling() {
  computeTimer_.reset();
  executeTimer_.reset();
}

const Timer &Region::getComputeTimer() const { return computeTimer_; }

const Timer &Region::getExecuteTimer() const { return executeTimer_; }

bool Region::operator==(const Region &o) const {

  if (initialized_ != o.initialized_ || outputs_.size() != o.outputs_.size() ||
      inputs_.size() != o.inputs_.size()) {
    return false;
  }

  if (name_ != o.name_ || type_ != o.type_ || dims_ != o.dims_ ||
      spec_ != o.spec_ || phases_ != o.phases_ ||
      dimensionInfo_ != o.dimensionInfo_) {
    return false;
  }

  // Compare Regions's Input
  static auto compareInput = [](decltype(*inputs_.begin()) a, decltype(a) b) {
    if (a.first != b.first ||
        a.second->isRegionLevel() != b.second->isRegionLevel()) {
      return false;
    }
    auto links1 = a.second->getLinks();
    auto links2 = b.second->getLinks();
    if (links1.size() != links2.size()) {
      return false;
    }
    for (size_t i = 0; i < links1.size(); i++) {
      if (*(links1[i]) != *(links2[i])) {
        return false;
      }
    }
    return true;
  };
  if (!std::equal(inputs_.begin(), inputs_.end(), o.inputs_.begin(),
                  compareInput)) {
    return false;
  }
  // Compare Regions's Output
  static auto compareOutput = [](decltype(*outputs_.begin()) a, decltype(a) b) {
    if (a.first != b.first ||
        a.second->isRegionLevel() != b.second->isRegionLevel() ||
        a.second->getNodeOutputElementCount() !=
            b.second->getNodeOutputElementCount()) {
      return false;
    }
    return true;
  };
  if (!std::equal(outputs_.begin(), outputs_.end(), o.outputs_.begin(),
                  compareOutput)) {
    return false;
  }

  return true;
}





// Internal methods called by RegionImpl.

Output *Region::getOutput(const std::string &name) const {
  auto o = outputs_.find(name);
  if (o == outputs_.end())
    return nullptr;
  return o->second;
}

Input *Region::getInput(const std::string &name) const {
  auto i = inputs_.find(name);
  if (i == inputs_.end())
    return nullptr;
  return i->second;
}

// Called by Network during serialization
const std::map<std::string, Input *> &Region::getInputs() const {
  return inputs_;
}

const std::map<std::string, Output *> &Region::getOutputs() const {
  return outputs_;
}


const Array& Region::getOutputData(const std::string &outputName) const {
  auto oi = outputs_.find(outputName);
  if (oi == outputs_.end())
    NTA_THROW << "getOutputData -- unknown output '" << outputName
              << "' on region " << getName();

  const Array& data = oi->second->getData();
  return data;
}

const Array& Region::getInputData(const std::string &inputName) const {
  auto ii = inputs_.find(inputName);
  if (ii == inputs_.end())
    NTA_THROW << "getInput -- unknown input '" << inputName << "' on region "
              << getName();

  const Array & data = ii->second->getData();
  return data;
}

void Region::prepareInputs() {
  // Ask each input to prepare itself
  for (InputMap::const_iterator i = inputs_.begin(); i != inputs_.end(); i++) {
    i->second->prepare();
  }
}


// setParameter

void Region::setParameterInt32(const std::string &name, Int32 value) {
  impl_->setParameterInt32(name, (Int64)-1, value);
}

void Region::setParameterUInt32(const std::string &name, UInt32 value) {
  impl_->setParameterUInt32(name, (Int64)-1, value);
}

void Region::setParameterInt64(const std::string &name, Int64 value) {
  impl_->setParameterInt64(name, (Int64)-1, value);
}

void Region::setParameterUInt64(const std::string &name, UInt64 value) {
  impl_->setParameterUInt64(name, (Int64)-1, value);
}

void Region::setParameterReal32(const std::string &name, Real32 value) {
  impl_->setParameterReal32(name, (Int64)-1, value);
}

void Region::setParameterReal64(const std::string &name, Real64 value) {
  impl_->setParameterReal64(name, (Int64)-1, value);
}

void Region::setParameterBool(const std::string &name, bool value) {
  impl_->setParameterBool(name, (Int64)-1, value);
}

// getParameter

Int32 Region::getParameterInt32(const std::string &name) const {
  return impl_->getParameterInt32(name, (Int64)-1);
}

Int64 Region::getParameterInt64(const std::string &name) const {
  return impl_->getParameterInt64(name, (Int64)-1);
}

UInt32 Region::getParameterUInt32(const std::string &name) const {
  return impl_->getParameterUInt32(name, (Int64)-1);
}

UInt64 Region::getParameterUInt64(const std::string &name) const {
  return impl_->getParameterUInt64(name, (Int64)-1);
}

Real32 Region::getParameterReal32(const std::string &name) const {
  return impl_->getParameterReal32(name, (Int64)-1);
}

Real64 Region::getParameterReal64(const std::string &name) const {
  return impl_->getParameterReal64(name, (Int64)-1);
}

bool Region::getParameterBool(const std::string &name) const {
  return impl_->getParameterBool(name, (Int64)-1);
}

// array parameters

void Region::getParameterArray(const std::string &name, Array &array) const {
  impl_->getParameterArray(name, (Int64)-1, array);
}

void Region::setParameterArray(const std::string &name, const Array &array) {
  impl_->setParameterArray(name, (Int64)-1, array);
}

void Region::setParameterString(const std::string &name, const std::string &s) {
  impl_->setParameterString(name, (Int64)-1, s);
}

std::string Region::getParameterString(const std::string &name) {
  return impl_->getParameterString(name, (Int64)-1);
}

bool Region::isParameterShared(const std::string &name) const {
  return impl_->isParameterShared(name);
}

} // namespace nupic
