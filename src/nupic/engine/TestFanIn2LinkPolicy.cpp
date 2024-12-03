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

#include <string>

#include <nupic/engine/Link.hpp>
#include <nupic/engine/TestFanIn2LinkPolicy.hpp>
#include <nupic/ntypes/Dimensions.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
TestFanIn2LinkPolicy::TestFanIn2LinkPolicy(const std::string params, Link *link)
    : link_(link), initialized_(false) {}

TestFanIn2LinkPolicy::~TestFanIn2LinkPolicy() {
  // We don't own link_ -- it is a reference to our parent.
}

void TestFanIn2LinkPolicy::setSrcDimensions(Dimensions &dims) {

  // This method should never be called if we've already been set
  NTA_CHECK(srcDimensions_.isUnspecified())
      << "Internal error on link " << link_->toString();
  NTA_CHECK(destDimensions_.isUnspecified())
      << "Internal error on link " << link_->toString();

  if (dims.isUnspecified())
    NTA_THROW << "Invalid unspecified source dimensions for link "
              << link_->toString();

  if (dims.isDontcare())
    NTA_THROW << "Invalid dontcare source dimensions for link "
              << link_->toString();

  // Induce destination dimensions from src dimensions based on a fan-in of 2
  Dimensions destDims;
  for (size_t i = 0; i < dims.size(); i++) {
    destDims.push_back(dims[i] / 2);
    if (destDims[i] * 2 != dims[i])
      NTA_THROW << "Invalid source dimensions " << dims.toString()
                << " for link " << link_->toString()
                << ". Dimensions must be multiples of 2";
  }

  srcDimensions_ = dims;
  destDimensions_ = destDims;
}

void TestFanIn2LinkPolicy::setDestDimensions(Dimensions &dims) {
  // This method should never be called if we've already been set
  NTA_CHECK(srcDimensions_.isUnspecified())
      << "Internal error on link " << link_->toString();
  NTA_CHECK(destDimensions_.isUnspecified())
      << "Internal error on link " << link_->toString();

  if (dims.isUnspecified())
    NTA_THROW << "Invalid unspecified dest dimensions for link "
              << link_->toString();

  if (dims.isDontcare())
    NTA_THROW << "Invalid dontcare dest dimensions for link "
              << link_->toString();

  Dimensions srcDims;
  for (size_t i = 0; i < dims.size(); i++) {
    // Induce src dimensions from destination dimensions based on a fan-in of 2
    // from src to dest which looks like fan-out of 2 from dest to src
    srcDims.push_back(dims[i] * 2);
  }

  srcDimensions_ = srcDims;
  destDimensions_ = dims;
}

const Dimensions &TestFanIn2LinkPolicy::getSrcDimensions() const {
  return srcDimensions_;
}

const Dimensions &TestFanIn2LinkPolicy::getDestDimensions() const {
  return destDimensions_;
}

void TestFanIn2LinkPolicy::setNodeOutputElementCount(size_t elementCount) {
  elementCount_ = elementCount;
}

void TestFanIn2LinkPolicy::buildProtoSplitterMap(
    Input::SplitterMap &splitter) const {

  NTA_CHECK(isInitialized());
  // node [i, j] in the source region sends data to node [i/2, j/2] in the dest
  // region. For N dimensions, this is naturally done as N nested loops. Do just
  // for N=1,2 for now
  if (srcDimensions_.size() == 1) {
    for (size_t i = 0; i < srcDimensions_[0]; i++) {
      splitter[i / 2].push_back(i);
    }
  } else if (srcDimensions_.size() == 2) {
    for (size_t y = 0; y < srcDimensions_[1]; y++) {
      for (size_t x = 0; x < srcDimensions_[0]; x++) {
        size_t srcIndex = srcDimensions_.getIndex(Dimensions(x, y));
        size_t destIndex = destDimensions_.getIndex(Dimensions(x / 2, y / 2));

        size_t baseOffset = srcIndex * elementCount_;
        for (size_t element = 0; element < elementCount_; element++) {
          splitter[destIndex].push_back(baseOffset + element);
        }
      }
    }
  } else {
    NTA_THROW << "TestFanIn2 link policy does not support "
              << srcDimensions_.size() << "-dimensional topologies. FIXME!";
  }
}

void TestFanIn2LinkPolicy::initialize() { initialized_ = true; }

bool TestFanIn2LinkPolicy::isInitialized() const { return initialized_; }

} // namespace nupic
