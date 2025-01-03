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
 * Implementation of Output class
 *
 */

#include <cstring>               // memset
#include <nupic/engine/Link.hpp> // temporary
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/BasicType.hpp>

namespace nupic {

Output::Output(Region &region, NTA_BasicType type, bool isRegionLevel,
               bool isSparse)
    : region_(region), isRegionLevel_(isRegionLevel), name_("Unnamed"),
      nodeOutputElementCount_(0), isSparse_(isSparse) {
  data_ = new Array(type);
}

Output::~Output() noexcept(false) {
  // If we have any outgoing links, then there has been an
  // error in the shutdown process. Not good to thow an exception
  // from a destructor, but we need to catch this error, and it
  // should never occur if nupic internal logic is correct.
  NTA_CHECK(links_.size() == 0) << "Internal error in region deletion";
  delete data_;
}

// allocate buffer
void Output::initialize(size_t count) {
  // reinitialization is ok
  // might happen if initial initialization failed with an
  // exception (elsewhere) and was retried.
  if (data_->getBuffer() != nullptr)
    return;

  if (isSparse_) {
    NTA_CHECK(isRegionLevel_) << "Sparse data must be region level";
    NTA_CHECK(data_->getType() == NTA_BasicType_UInt32)
        << "Sparse data must be uint32";
  }

  nodeOutputElementCount_ = count;
  size_t dataCount;
  if (isRegionLevel_)
    dataCount = count;
  else
    dataCount = count * region_.getDimensions().getCount();
  if (dataCount != 0) {
    data_->allocateBuffer(dataCount);
    // Zero the buffer because unitialized outputs can screw up inspectors,
    // which look at the output before compute(). NPC-60
    void *buffer = data_->getBuffer();
    size_t byteCount = dataCount * BasicType::getSize(data_->getType());
    memset(buffer, 0, byteCount);
  }
}

void Output::addLink(Link *link) {
  // Make sure we don't add the same link twice
  // It is a logic error if we add the same link twice here, since
  // this method should only be called from Input::addLink
  auto linkIter = links_.find(link);
  NTA_CHECK(linkIter == links_.end());

  links_.insert(link);
}

void Output::removeLink(Link *link) {
  auto linkIter = links_.find(link);
  // Should only be called internally. Logic error if link not found
  NTA_CHECK(linkIter != links_.end());
  // Output::removeLink is only called from Input::removeLink so we don't
  // have to worry about removing it on the Input side
  links_.erase(linkIter);
}

const Array &Output::getData() const { return *data_; }

bool Output::isRegionLevel() const { return isRegionLevel_; }

Region &Output::getRegion() const { return region_; }
bool Output::isSparse() const { return isSparse_; }

void Output::setName(const std::string &name) { name_ = name; }

const std::string &Output::getName() const { return name_; }

size_t Output::getNodeOutputElementCount() const {
  return nodeOutputElementCount_;
}

bool Output::hasOutgoingLinks() { return (!links_.empty()); }

NTA_BasicType Output::getDataType() const { return data_->getType(); }

} // namespace nupic
