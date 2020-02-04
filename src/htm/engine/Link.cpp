/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2017, Numenta, Inc.
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
 * Implementation of the Link class
 */
#include <cstring> // memcpy,memset
#include <htm/engine/Input.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/utils/Log.hpp>

// By calling  Network::setLogLevel(LogLevel_Verbose)
// you can enable the NTA_DEBUG macros below.

namespace htm {


Link::Link(const std::string &linkType, const std::string &linkParams,
           const std::string &srcRegionName, const std::string &destRegionName,
           const std::string &srcOutputName, const std::string &destInputName,
           const size_t propagationDelay) {
  commonConstructorInit_(linkType, linkParams, srcRegionName, destRegionName,
                         srcOutputName, destInputName, propagationDelay);
}

Link::Link(const std::string &linkType, const std::string &linkParams,
           std::shared_ptr<Output> srcOutput, std::shared_ptr<Input> destInput, const size_t propagationDelay) {
  commonConstructorInit_(linkType, linkParams, srcOutput->getRegion()->getName(),
                         destInput->getRegion()->getName(), srcOutput->getName(),
                         destInput->getName(), propagationDelay);

  connectToNetwork(srcOutput, destInput);
  // Note -- link is not usable until we set the destOffset, which happens at
  // initialization time
}

Link::Link() {  // needed for deserialization
  destOffset_ = 0;
  initialized_ = false;
}

void Link::commonConstructorInit_(const std::string &linkType,
                                  const std::string &linkParams,
                                  const std::string &srcRegionName,
                                  const std::string &destRegionName,
                                  const std::string &srcOutputName,
                                  const std::string &destInputName,
                                  const size_t propagationDelay) {
  linkType_ = linkType;
  linkParams_ = linkParams;
  srcRegionName_ = srcRegionName;
  srcOutputName_ = srcOutputName;
  destRegionName_ = destRegionName;
  destInputName_ = destInputName;
  propagationDelay_ = propagationDelay;
  destOffset_ = 0;
  is_FanIn_ = false;
  initialized_ = false;

}




void Link::initialize(size_t destinationOffset, bool is_FanIn) {
  // Make sure all information is specified and
  // consistent. Unless there is a NuPIC implementation
  // error, all these checks are guaranteed to pass
  // because of the way the network is constructed
  // and initialized.

  // Make sure we have been attached to a real network
  NTA_CHECK(src_)
      << "Link::initialize() and src_ Output object not set.";
  NTA_CHECK(dest_)
      << "Link::initialize() and dest_ Input object not set.";
 
  destOffset_ = destinationOffset;
  is_FanIn_ = is_FanIn;

  // ---
  // Initialize the propagation delay buffer
  // But skip it if it already has something in it from deserialize().
  // ---
  if (propagationDelay_ > 0 && propagationDelayBuffer_.empty()) {
    // Initialize delay data elements.  This must be done during initialize()
    // because the buffer size is not known prior to then.
    // front of queue will be the next value to be copied to the dest Input buffer.
    // back of queue will be the same as the current contents of source Output.
    Array &output_buffer = src_->getData();
    for (size_t i = 0; i < (propagationDelay_); i++) {
      Array delayedbuffer = output_buffer.copy();
      delayedbuffer.zeroBuffer();
      propagationDelayBuffer_.push_back(delayedbuffer);
    }
  }

  initialized_ = true;
}


// Return constructor params
const std::string &Link::getLinkType() const { return linkType_; }

const std::string &Link::getLinkParams() const { return linkParams_; }

const std::string &Link::getSrcRegionName() const { return srcRegionName_; }

const std::string &Link::getSrcOutputName() const { return srcOutputName_; }

const std::string &Link::getDestRegionName() const { return destRegionName_; }

const std::string &Link::getDestInputName() const { return destInputName_; }

std::string Link::getMoniker() const {
  std::stringstream ss;
  ss << getSrcRegionName() << "." << getSrcOutputName() << "-->"
     << getDestRegionName() << "." << getDestInputName();
  return ss.str();
}

const std::string Link::toString() const {
  std::stringstream ss;
  ss << "{" << getSrcRegionName() << "." << getSrcOutputName();
  if (src_) {
    ss <<  src_->getDimensions().toString();
  }
  ss << " to " << getDestRegionName() << "." << getDestInputName();
  if (dest_) {
    ss << dest_->getDimensions().toString();
  }
  return ss.str();
}

void Link::connectToNetwork(std::shared_ptr<Output> src, std::shared_ptr<Input> dest) {
  NTA_CHECK(src != nullptr);
  NTA_CHECK(dest != nullptr);

  src_ = src.get();
  dest_ = dest.get();
}

// The methods below only work on connected links.
Output* Link::getSrc() const

{
  NTA_CHECK(src_)
      << "Link::getSrc() can only be called on a connected link";
  return src_;
}

Input* Link::getDest() const {
  NTA_CHECK(dest_)
      << "Link::getDest() can only be called on a connected link";
  return dest_;
}


void Link::compute() {
  NTA_CHECK(initialized_);

  if (propagationDelay_) {
    // A delayed link's queue buffer size should always be number of delays.
    NTA_CHECK(propagationDelayBuffer_.size() == (propagationDelay_));
  }

  // Copy data from source to destination. For delayed links, will copy from
  // head of circular queue; otherwise directly from source.
  const Array &src = propagationDelay_ ? propagationDelayBuffer_.front() : src_->getData();
  Array &dest = dest_->getData();

  NTA_DEBUG << "compute Link: copying " << getMoniker()
              << "; delay=" << propagationDelay_ << "; size=" << src.getCount()
              << " type=" << BasicType::getName(src.getType())
              << " --> " << BasicType::getName(dest.getType()) << std::endl;

	NTA_CHECK(src.getCount() + destOffset_ <= dest.getCount())
        << "Not enough room in buffer to propogate to " << destRegionName_
        << " " << destInputName_ << ". ";

  if (src.getType() == dest.getType() && !is_FanIn_ && propagationDelay_==0) {
    dest = src;   // Performs a shallow copy. Data not copied but passed in shared_ptr.
  } else {
    // we must perform a deep copy with possible type conversion.
    // It is copied into the destination Input
    // buffer at the specified offset so an Input with multiple incoming links
    // has the Output buffers appended into a single large Input buffer.
    src.convertInto(dest, destOffset_, dest.getCount());
  }
}

void Link::shiftBufferedData() {
  if (propagationDelay_) {   // Source buffering is not used in 0-delay links
    Array& from = src_->getData();
    NTA_CHECK(propagationDelayBuffer_.size() == (propagationDelay_));

    // push a copy of the source Output buffer on the back of the queue.
    // This must be a deep copy.
    Array a = from.copy();
    propagationDelayBuffer_.push_back(a);

    // Pop the head of the queue
    // The top of the queue now becomes the value to copy to destination.
    propagationDelayBuffer_.pop_front();
  }
}

std::deque<Array> Link::preSerialize() const {
  std::deque<Array> delay;
  if (propagationDelay_ > 0) {
    // we need to capture the propagationDelayBuffer_ used for propagationDelay
    // Do not copy the last entry.  It is the same as the output buffer.

    // The current contents of the Destination Input buffer also needs
    // to be captured as if it were the top value of the propagationDelayBuffer.
    // When restored, it will be copied to the dest input buffer and popped off
    // before the next execution. If there is a fanIn, we only
    // want to capture the amount of the input buffer contributed by
    // this link.
    size_t srcCount = 0;
    if (src_) {
      const Array& s = src_->getData();
      srcCount = s.getCount();
    }
    Array a = dest_->getData().subset(destOffset_, srcCount);
    delay.push_back(a); // our part of the current Dest Input buffer.

    for (auto itr = propagationDelayBuffer_.begin();
          itr != propagationDelayBuffer_.end(); itr++) {
      if (itr + 1 == propagationDelayBuffer_.end())
        break; // skip the last buffer. Its the current output.
      delay.push_back(*itr);
    } // end for
  }
  return delay;
}

bool Link::operator==(const Link &o) const {
  if (initialized_ != o.initialized_ ||
      propagationDelay_ != o.propagationDelay_ || 
      linkType_ != o.linkType_ ||
      linkParams_ != o.linkParams_ || 
      destOffset_ != o.destOffset_ ||
      is_FanIn_ != o.is_FanIn_ ||
      srcRegionName_ != o.srcRegionName_ ||
      destRegionName_ != o.destRegionName_ ||
      srcOutputName_ != o.srcOutputName_ ||
      destInputName_ != o.destInputName_) {
    return false;
	// TODO: compare propagationDelayBuffer
  }
  return true;
}


/**
 * A readable display of a Link.
 * This is not part of the save/load facility.
 */
std::ostream &operator<<(std::ostream &f, const Link &link) {
  f << "Link: {\n";
  f << "  src: " << link.getSrcRegionName() << "." << link.getSrcOutputName() << ",\n";
  f << "  dest: " << link.getDestRegionName() << "." << link.getDestInputName() << ",\n";
  f << "  type: " << link.getLinkType() << ",  params: " << link.getLinkParams() << ",\n";
  f << "  fanIn: " << link.is_FanIn_ << ",  offset: " << link.destOffset_ << ",\n";
  f << "  propagationDelay: " << link.getPropagationDelay()<< ",\n";
  if (link.getPropagationDelay() > 0) {
  	f <<   "   [\n";
	  for (auto buf : link.propagationDelayBuffer_) {
		  f << "    " << buf << "\n";
	  }
	  f <<   "   ]\n";
  }
  f << "}\n";
  return f;
}

} // namespace htm
