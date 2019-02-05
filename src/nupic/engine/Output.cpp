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
 * Implementation of Output class
 *
 */

#include <cstring>               // memset
#include <nupic/engine/Link.hpp> // temporary
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/types/BasicType.hpp>

namespace nupic {

Output::Output(Region* region, NTA_BasicType type, bool isRegionLevel)
    : region_(region), isRegionLevel_(isRegionLevel), name_("Unnamed"),
      nodeOutputElementCount_(0) {
  data_ = Array(type);
}

Output::~Output() noexcept(false) {
  // If we have any outgoing links, then there has been an
  // error in the shutdown process. Not good to thow an exception
  // from a destructor, but we need to catch this error, and it
  // should never occur if nupic internal logic is correct.
  NTA_CHECK(links_.size() == 0) << "Internal error in region deletion, still has links.";
}

// allocate buffer
// The 'count' argument comes from the impl by calling getNodeOutputElementCount()
void Output::initialize(size_t count) {
  // reinitialization is ok
  // might happen if initial initialization failed with an
  // exception (elsewhere) and was retried.
  if (data_.has_buffer())
    return;

  nodeOutputElementCount_ = count;
  size_t dataCount;
  if (isRegionLevel_)
    dataCount = count;
  else
    dataCount = count * region_->getDimensions().getCount();
  if (dataCount != 0) {
    if (data_.getType() == NTA_BasicType_SDR && isRegionLevel_) {
      const vector<UInt> dim = region_->getDimensions();
      data_.allocateBuffer(dim);
    } else {
      data_.allocateBuffer(dataCount);
      // Zero the buffer because unitialized outputs can screw up inspectors,
      // which look at the output before compute(). NPC-60
      data_.zeroBuffer();
    }
  }
}

void Output::addLink(std::shared_ptr<Link> link) {
  // Make sure we don't add the same link twice
  // It is a logic error if we add the same link twice here, since
  // this method should only be called from Input::addLink
  auto linkIter = links_.find(link);
  NTA_CHECK(linkIter == links_.end());

  links_.insert(link);
}

void Output::removeLink(std::shared_ptr<Link> link) {
  auto linkIter = links_.find(link);
  // Should only be called internally. Logic error if link not found
  NTA_CHECK(linkIter != links_.end());
  // Output::removeLink is only called from Input::removeLink so we don't
  // have to worry about removing it on the Input side
  links_.erase(linkIter);
}



bool Output::isRegionLevel() const { return isRegionLevel_; }

Region* Output::getRegion() const { return region_; }

void Output::setName(const std::string &name) { name_ = name; }

const std::string &Output::getName() const { return name_; }

size_t Output::getNodeOutputElementCount() const {
  return nodeOutputElementCount_;
}

bool Output::hasOutgoingLinks() { return (!links_.empty()); }

NTA_BasicType Output::getDataType() const { return data_.getType(); }

} // namespace nupic
