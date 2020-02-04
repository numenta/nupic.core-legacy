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
 * Implementation of Output class
 *
 */

#include <htm/engine/Link.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/engine/Region.hpp>

using namespace htm;

Output::Output(Region* region, const std::string& outputName, NTA_BasicType type)
    : region_(region),
      name_(outputName) {
  data_ = Array(type);
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
      data_.allocateBuffer(dim_.asVector());
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

  const std::shared_ptr<Spec>& srcSpec = region_->getSpec();

  if (!dim_.isSpecified()) {
    dim_.clear();
    // ask the spec how big the buffer is.
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
  }

  // If we still have a isDontcare, check if the spec defines
  // regionLevel then get the dimensions from the region dims.
  bool regionLevel = srcSpec->outputs.getByName(name_).regionLevel;
  if (regionLevel) {
    Dimensions d = region_->getDimensions();
    if (dim_.isDontcare()&& d.isSpecified()) {
      dim_ = d;
    }
    else if (dim_.isSpecified() && !d.isSpecified()) {
      region_->setDimensions(dim_);
    }
  }
  return dim_;
}

void Output::addLink(const std::shared_ptr<Link> link) {
  // Make sure we don't add the same link twice
  // It is a logic error if we add the same link twice here, since
  // this method should only be called from Input::addLink
  const auto linkIter = links_.find(link);
  NTA_CHECK(linkIter == links_.end());

  links_.insert(link);
}

void Output::removeLink(const std::shared_ptr<Link>& link) {
  // Should only be called internally. Logic error if link not found
  const auto linkIter = links_.find(link);
  NTA_CHECK(linkIter != links_.end()) << "Link not found.";
  // Output::removeLink is only called from Input::removeLink so we don't
  // have to worry about removing it on the Input side
  links_.erase(linkIter);
}

namespace htm {
  std::ostream &operator<<(std::ostream &f, const Output &d) {
    f << "Output: " << d.getRegion()->getName() << "." << d.getName() << " " << d.getData();
    return f;
  }
}

Region* Output::getRegion() const { return region_; }

void Output::setName(const std::string &name) { name_ = name; }

const std::string &Output::getName() const { return name_; }

size_t Output::getNodeOutputElementCount() const {
  return dim_.getCount();
}

bool Output::hasOutgoingLinks() { return (!links_.empty()); }

NTA_BasicType Output::getDataType() const { return data_.getType(); }

void Output::resize(size_t count) {
  NTA_CHECK(data_.getType() != NTA_BasicType_SDR) << "Cannot resize SDR buffer.";
  NTA_CHECK(data_.getType() != NTA_BasicType_Str) << "Cannot resize Str buffer.";
  // Allocate a new buffer and copy previous data.
  Array d(data_.getType());
  d.allocateBuffer(count);
  size_t w = BasicType::getSize(data_.getType());
  char *p = reinterpret_cast<char *>(data_.getBuffer());
  char *s = reinterpret_cast<char *>(d.getBuffer());
  size_t i;
  for (i = 0; i < data_.getCount() * w; i++) {
    if (i < count)
      s[i] = p[i];
  }
  for (; i < count * w; i++) {
    s[i] = 0;
  }
  data_ = d;
  dim_ = {static_cast<UInt32>(count)};

  // If the output is resized then the inputs to which it is connected
  // must also be resized.  If the input is a Fan-in, the offsets into
  // that buffer must be adjusted.
  for (auto link : links_) {
    Input *dest = link->getDest();
    dest->resize();
  }
}


