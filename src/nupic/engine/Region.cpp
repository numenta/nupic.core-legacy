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
Implementation of the Region class

Methods related to parameters are in Region_parameters.cpp
Methods related to inputs and outputs are in Region_io.cpp

*/

#include <iostream>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Link.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/NodeSet.hpp>
#include <nupic/os/Timer.hpp>
#include <nupic/proto/RegionProto.capnp.h>
#include <nupic/utils/Log.hpp>
#include <set>
#include <stdexcept>
#include <string>

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

  impl_ = factory.createRegionImpl(nodeType, nodeParams, this);
  createInputsAndOutputs_();
}

// Deserialize region
Region::Region(std::string name, const std::string &nodeType,
               const Dimensions &dimensions, BundleIO &bundle, Network *network)
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
    if (!dimensions.isDontcare() && !dimensions.isUnspecified() &&
        !dimensions.isOnes())
      NTA_THROW << "Attempt to deserialize region of type " << nodeType
                << " with dimensions " << dimensions
                << " but region supports exactly one node.";

  dims_ = dimensions;

  impl_ = factory.deserializeRegionImpl(nodeType, bundle, this);
  createInputsAndOutputs_();
}

Region::Region(std::string name, RegionProto::Reader &proto, Network *network)
    : name_(std::move(name)), type_(proto.getNodeType().cStr()),
      initialized_(false), enabledNodes_(nullptr), network_(network),
      profilingEnabled_(false) {
  read(proto);
  createInputsAndOutputs_();
}

Network *Region::getNetwork() { return network_; }

void Region::createInputsAndOutputs_() {

  // Create all the outputs for this node type. By default outputs are zero size
  for (size_t i = 0; i < spec_->outputs.getCount(); ++i) {
    const std::pair<std::string, OutputSpec> &p = spec_->outputs.getByIndex(i);
    std::string outputName = p.first;
    const OutputSpec &os = p.second;
    auto output = new Output(*this, os.dataType, os.regionLevel, os.sparse);
    outputs_[outputName] = output;
    // keep track of name in the output also -- see note in Region.hpp
    output->setName(outputName);
  }

  // Create all the inputs for this node type.
  for (size_t i = 0; i < spec_->inputs.getCount(); ++i) {
    const std::pair<std::string, InputSpec> &p = spec_->inputs.getByIndex(i);
    std::string inputName = p.first;
    const InputSpec &is = p.second;

    auto input = new Input(*this, is.dataType, is.regionLevel, is.sparse);
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
  // If there are any links connected to our outputs, this will fail.
  // We should catch this error in the Network class and give the
  // user a good error message (regions may be removed either in
  // Network::removeRegion or Network::~Network())
  for (auto &elem : outputs_) {
    delete elem.second;
    elem.second = nullptr;
  }

  for (auto &elem : inputs_) {
    delete elem.second;
    elem.second = nullptr;
  }

  delete impl_;
  delete enabledNodes_;
}

void Region::initialize() {

  if (initialized_)
    return;

  impl_->initialize();
  initialized_ = true;
}

bool Region::isInitialized() const { return initialized_; }

const std::string &Region::getName() const { return name_; }

const std::string &Region::getType() const { return type_; }

const Spec *Region::getSpec() const { return spec_; }

const Spec *Region::getSpecFromType(const std::string &nodeType) {
  RegionImplFactory &factory = RegionImplFactory::getInstance();
  return factory.getSpec(nodeType);
}

void Region::registerPyRegion(const std::string module,
                              const std::string className) {
  RegionImplFactory::registerPyRegion(module, className);
}

void Region::registerCPPRegion(const std::string name,
                               GenericRegisteredRegionImpl *wrapper) {
  RegionImplFactory::registerCPPRegion(name, wrapper);
}

void Region::unregisterPyRegion(const std::string className) {
  RegionImplFactory::unregisterPyRegion(className);
}

void Region::unregisterCPPRegion(const std::string name) {
  RegionImplFactory::unregisterCPPRegion(name);
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
  int nIncompleteLinks = 0;
  for (auto &elem : inputs_) {
    nIncompleteLinks += (elem.second)->evaluateLinks();
  }
  return nIncompleteLinks;
}

std::string Region::getLinkErrors() const {

  std::stringstream ss;
  for (const auto &elem : inputs_) {
    const std::vector<Link *> &links = elem.second->getLinks();
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
  NTA_CHECK(spec_->outputs.contains(name));
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
  // Some outputs are optional. These outputs will have 0 elementCount in the
  // node spec and also return 0 from impl->getNodeOutputElementCount(). These
  // outputs still appear in the output map, but with an array size of 0.

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

  // can only create the enabled node set after we know the number of dimensions
  setupEnabledNodeSet();
}

void Region::setupEnabledNodeSet() {
  NTA_CHECK(dims_.isValid());

  if (enabledNodes_ != nullptr) {
    delete enabledNodes_;
  }

  size_t nnodes = dims_.getCount();
  enabledNodes_ = new NodeSet(nnodes);

  enabledNodes_->allOn();
}

const NodeSet &Region::getEnabledNodes() const {
  if (enabledNodes_ == nullptr) {
    NTA_THROW << "Attempt to access enabled nodes set before region has been "
                 "initialized";
  }
  return *enabledNodes_;
}

void Region::setDimensionInfo(const std::string &info) {
  dimensionInfo_ = info;
}

const std::string &Region::getDimensionInfo() const { return dimensionInfo_; }

void Region::removeAllIncomingLinks() {
  InputMap::const_iterator i = inputs_.begin();
  for (; i != inputs_.end(); i++) {
    std::vector<Link *> links = i->second->getLinks();
    for (auto &links_link : links) {
      i->second->removeLink(links_link);
    }
  }
}

void Region::uninitialize() { initialized_ = false; }

void Region::setPhases(std::set<UInt32> &phases) { phases_ = phases; }

std::set<UInt32> &Region::getPhases() { return phases_; }

void Region::serializeImpl(BundleIO &bundle) { impl_->serialize(bundle); }

void Region::write(RegionProto::Builder &proto) const {
  auto dimensionsProto = proto.initDimensions(dims_.size());
  for (UInt i = 0; i < dims_.size(); ++i) {
    dimensionsProto.set(i, dims_[i]);
  }
  auto phasesProto = proto.initPhases(phases_.size());
  UInt i = 0;
  for (auto elem : phases_) {
    phasesProto.set(i++, elem);
  }
  proto.setNodeType(type_.c_str());
  auto implProto = proto.getRegionImpl();
  impl_->write(implProto);
}

void Region::read(RegionProto::Reader &proto) {
  dims_.clear();
  for (auto elem : proto.getDimensions()) {
    dims_.push_back(elem);
  }

  phases_.clear();
  for (auto elem : proto.getPhases()) {
    phases_.insert(elem);
  }

  auto implProto = proto.getRegionImpl();
  RegionImplFactory &factory = RegionImplFactory::getInstance();
  spec_ = factory.getSpec(type_);
  impl_ = factory.deserializeRegionImpl(proto.getNodeType().cStr(), implProto,
                                        this);
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
        a.second->isRegionLevel() != b.second->isRegionLevel() ||
        a.second->isSparse() != b.second->isSparse()) {
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
        a.second->isSparse() != b.second->isSparse() ||
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
} // namespace nupic

} // namespace nupic
