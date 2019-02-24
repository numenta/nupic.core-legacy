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

#include <nupic/engine/Link.hpp> 
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/types/BasicType.hpp>

namespace nupic {

Output::Output(Region* region, const std::string& outputName, NTA_BasicType type)
    : region_(region), 
      name_(outputName) {
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
void Output::initialize() {
  // reinitialization is ok
  // might happen if initial initialization failed with an
  // exception (elsewhere) and was retried.
  if (data_.has_buffer())
    return;


  NTA_CHECK(!dim_.isDontcare())
        << "Output Dimensions cannot be determined for Region "
        << region_->getName() << "; output " << name_;

  size_t count = dim_.getCount();
  if (data_.getType() == NTA_BasicType_SDR) {
      data_.allocateBuffer(dim_);
  } else {
    data_.allocateBuffer(count);
    // Zero the buffer because unitialized outputs can screw up inspectors,
    // which look at the output before compute(). NPC-60
    data_.zeroBuffer();
  }
}


Dimensions Output::determineDimensions() {
  if (data_.has_buffer())
    return dim_;

  if (!dim_.isSpecified()) {
    dim_.clear();
    // ask the spec how big the buffer is.
    const std::shared_ptr<Spec>& srcSpec = region_->getSpec();
    UInt32 count = (UInt32)srcSpec->outputs.getByName(name_).count; 
    if (count > 0) {
      dim_.push_back(count);
    } else {
      // ask the region impl what the output dimensions are.
      dim_ = region_->askImplForOutputDimensions(name_);
      if (dim_.isUnspecified()) {
        dim_.push_back(0);  // set Don't care.
      }
    }

    // If we still have a isDontcare, check if the spec defines 
    // an input on this region to get the dimensions from. 
    if (dim_.isDontcare()) {
      std::string inheritFrom = srcSpec->outputs.getByName(name_).inheritFrom;
      if (!inheritFrom.empty()) {
        Input* in = region_->getInput(inheritFrom);
        if (in) {
          Dimensions d = in->getDimensions();
          if (d.isSpecified()) {
            dim_ = d;
          }
        }
      }
    }
  }
  return dim_;
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


Region* Output::getRegion() const { return region_; }

void Output::setName(const std::string &name) { name_ = name; }

const std::string &Output::getName() const { return name_; }

size_t Output::getNodeOutputElementCount() const {
  return dim_.getCount();
}

bool Output::hasOutgoingLinks() { return (!links_.empty()); }

NTA_BasicType Output::getDataType() const { return data_.getType(); }

} // namespace nupic
