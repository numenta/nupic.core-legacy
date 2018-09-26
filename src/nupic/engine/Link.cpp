/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of the Link class
 */
#include <cstring> // memcpy,memset
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Link.hpp>
#include <nupic/engine/LinkPolicy.hpp>
#include <nupic/engine/LinkPolicyFactory.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/utils/Log.hpp>

// Set this to true when debugging to enable handy debug-level logging of data
// moving through the links, including the delayed link transitions.
#define _LINK_DEBUG false

namespace nupic {

// Represents 'zero' scalar value used to compare Input/Output buffer contents
// for non-zero values
const static NTA_Real64 ZERO_VALUE = 0;

Link::Link(const std::string &linkType, const std::string &linkParams,
           const std::string &srcRegionName, const std::string &destRegionName,
           const std::string &srcOutputName, const std::string &destInputName,
           const size_t propagationDelay)
    : srcBuffer_(0) {
  commonConstructorInit_(linkType, linkParams, srcRegionName, destRegionName,
                         srcOutputName, destInputName, propagationDelay);
}

Link::Link(const std::string &linkType, const std::string &linkParams,
           Output *srcOutput, Input *destInput, const size_t propagationDelay)
    : srcBuffer_(0) {
  commonConstructorInit_(linkType, linkParams, srcOutput->getRegion().getName(),
                         destInput->getRegion().getName(), srcOutput->getName(),
                         destInput->getName(), propagationDelay);

  connectToNetwork(srcOutput, destInput);
  // Note -- link is not usable until we set the destOffset, which happens at
  // initialization time
}

Link::Link() : srcBuffer_(0) {}

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
  srcOffset_ = 0;
  srcSize_ = 0;
  src_ = nullptr;
  dest_ = nullptr;
  initialized_ = false;

  impl_ = LinkPolicyFactory().createLinkPolicy(linkType, linkParams, this);
}

Link::~Link() { delete impl_; }

void Link::initPropagationDelayBuffer_(size_t propagationDelay,
                                       const Array &original) {
  if (srcBuffer_.capacity() != 0 || !propagationDelay) {
    // Already initialized(e.g., as result of deserialization); or a 0-delay
    // link, which doesn't use buffering.
    return;
  }

  // Establish capacity for the requested delay data elements
  srcBuffer_.set_capacity(propagationDelay);

  // Initialize delay data elements
  size_t dataBufferSize = original.getBufferSize();
  NTA_BasicType dataElementType = original.getType();
  size_t dataElementCount =
      dataBufferSize / BasicType::getSize(dataElementType);
  for (size_t i = 0; i < propagationDelay; i++) {
    Array arrayTemplate(dataElementType);

    srcBuffer_.push_back(arrayTemplate);

    // Allocate 0-initialized data for current element
    srcBuffer_[i].allocateBuffer(dataElementCount);
    ::memset(srcBuffer_[i].getBuffer(), 0, dataBufferSize);
    srcBuffer_[i].setCount(0);
  }
  if (_LINK_DEBUG) {
    NTA_DEBUG << "Initialzed PropagationDelayBuffer: " << "\n";
    NTA_DEBUG <<   "  <propagationDelayBuffer>\n";
    NTA_DEBUG <<   "    <capacity> " << srcBuffer_.capacity() << " </capacity>\n";
    NTA_DEBUG<<    "    <size> " << srcBuffer_.size() << " </size>\n";
    for (auto buf : srcBuffer_) {
      NTA_DEBUG << "    " << buf << "\n";
    }
    NTA_DEBUG <<   "  </propagationDelayBuffer>\n";
  }
}

void Link::initialize(size_t destinationOffset) {
  // Make sure all information is specified and
  // consistent. Unless there is a NuPIC implementation
  // error, all these checks are guaranteed to pass
  // because of the way the network is constructed
  // and initialized.

  // Make sure we have been attached to a real network
  NTA_CHECK(src_ != nullptr);
  NTA_CHECK(dest_ != nullptr);

  // Confirm that our dimensions are consistent with the
  // dimensions of the regions we're connecting.
  const Dimensions &srcD = getSrcDimensions();
  const Dimensions &destD = getDestDimensions();
  NTA_CHECK(!srcD.isUnspecified());
  NTA_CHECK(!destD.isUnspecified());

  Dimensions oneD;
  oneD.push_back(1);

  if (src_->isRegionLevel()) {
    Dimensions d;
    for (size_t i = 0; i < src_->getRegion().getDimensions().size(); i++) {
      d.push_back(1);
    }

    NTA_CHECK(srcD.isDontcare() || srcD == d);
  } else if (src_->getRegion().getDimensions() == oneD) {
    Dimensions d;
    for (size_t i = 0; i < srcD.size(); i++) {
      d.push_back(1);
    }
    NTA_CHECK(srcD.isDontcare() || srcD == d);
  } else {
    NTA_CHECK(srcD.isDontcare() || srcD == src_->getRegion().getDimensions());
  }

  if (dest_->isRegionLevel()) {
    Dimensions d;
    for (size_t i = 0; i < dest_->getRegion().getDimensions().size(); i++) {
      d.push_back(1);
    }

    NTA_CHECK(destD.isDontcare() || destD.isOnes());
  } else if (dest_->getRegion().getDimensions() == oneD) {
    Dimensions d;
    for (size_t i = 0; i < destD.size(); i++) {
      d.push_back(1);
    }
    NTA_CHECK(destD.isDontcare() || destD == d);
  } else {
    NTA_CHECK(destD.isDontcare() ||
              destD == dest_->getRegion().getDimensions());
  }

  // Validate sparse link
  if (src_->isSparse() && !dest_->isSparse()) {
    // Sparse to dense: unit32 -> bool
    NTA_CHECK(dest_->getDataType() == NTA_BasicType_Bool)
        << "Sparse to Dense link destination must be boolean";
  } else if (!src_->isSparse() && dest_->isSparse()) {
    // Dense to sparse:  NTA_BasicType -> uint32
    NTA_CHECK(dest_->getDataType() == NTA_BasicType_UInt32)
        << "Dense to Sparse link destination must be uint32";
  }

  destOffset_ = destinationOffset;
  impl_->initialize();

  // ---
  // Initialize the propagation delay buffer
  // ---
  initPropagationDelayBuffer_(propagationDelay_, src_->getData());

  initialized_ = true;
}

void Link::setSrcDimensions(Dimensions &dims) {
  NTA_CHECK(src_ != nullptr && dest_ != nullptr)
      << "Link::setSrcDimensions() can only be called on a connected link";

  size_t nodeElementCount = src_->getNodeOutputElementCount();
  if (nodeElementCount == 0) {
    nodeElementCount =
        src_->getRegion().getNodeOutputElementCount(src_->getName());
  }
  impl_->setNodeOutputElementCount(nodeElementCount);

  impl_->setSrcDimensions(dims);
}

void Link::setDestDimensions(Dimensions &dims) {
  NTA_CHECK(src_ != nullptr && dest_ != nullptr)
      << "Link::setDestDimensions() can only be called on a connected link";

  size_t nodeElementCount = src_->getNodeOutputElementCount();
  if (nodeElementCount == 0) {
    nodeElementCount =
        src_->getRegion().getNodeOutputElementCount(src_->getName());
  }
  impl_->setNodeOutputElementCount(nodeElementCount);

  impl_->setDestDimensions(dims);
}

const Dimensions &Link::getSrcDimensions() const {
  return impl_->getSrcDimensions();
};

const Dimensions &Link::getDestDimensions() const {
  return impl_->getDestDimensions();
};

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
  ss << "[" << getSrcRegionName() << "." << getSrcOutputName();
  if (src_) {
    ss << " (region dims: " << src_->getRegion().getDimensions().toString()
       << ") ";
  }
  ss << " to " << getDestRegionName() << "." << getDestInputName();
  if (dest_) {
    ss << " (region dims: " << dest_->getRegion().getDimensions().toString()
       << ") ";
  }
  ss << " type: " << linkType_ << "]";
  return ss.str();
}

void Link::connectToNetwork(Output *src, Input *dest) {
  NTA_CHECK(src != nullptr);
  NTA_CHECK(dest != nullptr);

  src_ = src;
  dest_ = dest;
}

// The methods below only work on connected links.
Output &Link::getSrc() const

{
  NTA_CHECK(src_ != nullptr)
      << "Link::getSrc() can only be called on a connected link";
  return *src_;
}

Input &Link::getDest() const {
  NTA_CHECK(dest_ != nullptr)
      << "Link::getDest() can only be called on a connected link";
  return *dest_;
}

void Link::buildSplitterMap(Input::SplitterMap &splitter) {
  // The link policy generates a splitter map
  // at the element level.  Here we convert it
  // to a full splitter map
  //
  // if protoSplitter[destNode][x] == srcElement for some x
  // means that the output srcElement is sent to destNode

  Input::SplitterMap protoSplitter;
  protoSplitter.resize(splitter.size());
  size_t nodeElementCount = src_->getNodeOutputElementCount();
  impl_->setNodeOutputElementCount(nodeElementCount);
  impl_->buildProtoSplitterMap(protoSplitter);

  for (size_t destNode = 0; destNode < splitter.size(); destNode++) {
    // convert proto-splitter values into real
    // splitter values;
    for (auto &elem : protoSplitter[destNode]) {
      size_t srcElement = elem;
      size_t elementOffset = srcElement + destOffset_;
      splitter[destNode].push_back(elementOffset);
    }
  }
}

void Link::compute() {
  NTA_CHECK(initialized_);

  if (propagationDelay_) {
    NTA_CHECK(!srcBuffer_.empty());
  }

  // Copy data from source to destination. For delayed links, will copy from
  // head of circular queue; otherwise directly from source.
  const Array &src = propagationDelay_ ? srcBuffer_[0] : src_->getData();

  const Array &dest = dest_->getData();

  size_t srcSize = src.getBufferSize();
  size_t typeSize = BasicType::getSize(src.getType());
  size_t destByteOffset = destOffset_ * typeSize;

  if (_LINK_DEBUG) {
    NTA_DEBUG << "Link::compute: " << getMoniker() << "; copying to dest input"
              << "; delay=" << propagationDelay_ << "; " << src.getCount()
              << " elements=" << src;
  }

  if (src_->isSparse() == dest_->isSparse()) {
    // No conversion required, just copy the buffer over
    ::memcpy((char *)(dest.getBuffer()) + destByteOffset, src.getBuffer(),
             srcSize);
    if (dest_->isSparse()) {
      // Remove 'const' to update the variable length array
      const_cast<Array &>(dest).setCount(src.getCount());
    }
  } else if (dest_->isSparse()) {
    // Destination is sparse, convert source from dense to sparse

    // Sparse Output must be NTA_UInt32. See "initialize".
    NTA_UInt32 *destBuf =
        (NTA_UInt32 *)((char *)(dest.getBuffer()) + destByteOffset);

    // Dense source can be any scalar type. The scalar values will be lost
    // and only the indexes of the non-zero values will be stored.
    char *srcBuf = (char *)src.getBuffer();
    size_t destLen = dest.getBufferSize();
    size_t destIdx = 0;
    for (size_t i = 0; i < srcSize; i++) {
      // Check for any non-zero scalar value
      if (::memcmp(srcBuf + i * typeSize, &ZERO_VALUE, typeSize)) {
        NTA_CHECK(destIdx < destLen) << "Link destination is too small. "
                                     << "It should be at least " << destIdx + 1;
        destBuf[destIdx++] = i;
      }
    }
    // Remove 'const' to update the variable length array
    const_cast<Array &>(dest).setCount(destIdx);
  } else {
    // Destination is dense, convert source from sparse to dense

    // Sparse Input must be NTA_UInt32. See "initialize".
    NTA_UInt32 *srcBuf = (NTA_UInt32 *)src.getBuffer();

    // Dense destination links must be bool. See "initialize".
    bool *destBuf = (bool *)((char *)dest.getBuffer() + destByteOffset);

    size_t srcLen = src.getCount();
    size_t destLen = dest.getBufferSize();
    ::memset(destBuf, 0, destLen);
    size_t destIdx;
    for (size_t i = 0; i < srcLen; i++) {
      destIdx = srcBuf[i];
      NTA_CHECK(destIdx < destLen) << "Link destination is too small. "
                                   << "It should be at least " << destIdx + 1;
      destBuf[destIdx] = true;
    }
  }
}

void Link::shiftBufferedData() {
  if (!propagationDelay_) {
    // Source buffering is not used in 0-delay links
    return;
  }

  // A delayed link's circular buffer should always be at capacity, because
  // it starts out full in link initialization and we always append the new
  // source value after shifting out the head.
  NTA_CHECK(srcBuffer_.full());

  // Pop head of circular queue

  if (_LINK_DEBUG) {
    NTA_DEBUG << "Link::shiftBufferedData: " << getMoniker()
              << "; popping head; " << srcBuffer_[0].getCount()
              << " elements=" << srcBuffer_[0];
  }

  //srcBuffer_.pop_front();  // the circular buffer automatically does this

  // Append the current src value to circular queue

  const Array &srcArray = src_->getData();
  size_t elementCount = srcArray.getCount();
  auto elementType = srcArray.getType();
  size_t bufferSize = srcArray.getBufferSize();
  size_t maxElementCount = bufferSize / BasicType::getSize(elementType);

  if (_LINK_DEBUG) {
    NTA_DEBUG << "Link::shiftBufferedData: " << getMoniker()
              << "; appending src to circular buffer; " << elementCount
              << " elements=" << srcArray;

    NTA_DEBUG << "Link::shiftBufferedData: " << getMoniker()
              << "; num arrays in circular buffer before append; "
              << srcBuffer_.size() << "; capacity=" << srcBuffer_.capacity();
  }

  Array array(elementType);
  srcBuffer_.push_back(array);

  auto &lastElement = srcBuffer_.back();
  lastElement.allocateBuffer(maxElementCount);
  ::memcpy(lastElement.getBuffer(), srcArray.getBuffer(),
           elementCount * BasicType::getSize(elementType));
  lastElement.setCount(elementCount);

  if (_LINK_DEBUG) {
    NTA_DEBUG << "Link::shiftBufferedData: " << getMoniker()
              << "; circular buffer head after append is: "
              << srcBuffer_[0].getCount() << " elements=" << srcBuffer_[0];
  }
}

void Link::serialize(std::ostream &f) {
  size_t srcCount = ((!src_) ? (size_t)0 : src_->getData().getCount());

  f << "{\n";
  f << "linkType: " <<  getLinkType() << "\n";
  f << "params: " << getLinkParams() << "\n";
  f << "srcRegion: " << getSrcRegionName() << "\n";
  f << "srcOutput: " << getSrcOutputName() << "\n";
  f << "destRegion: " << getDestRegionName() << "\n";
  f << "destInput: " << getDestInputName() << "\n";
  f << "propagationDelay: " << propagationDelay_ << "\n";
  f << "propagationDelayBuffer: [ " << srcBuffer_.size() << "\n";
  if (propagationDelay_ > 0) {
    // we need to capture the propagationDelayBuffer_ used for propagationDelay
    // Do not copy the last entry.  It is the same as the output buffer.

    // The current contents of the Destination Input buffer also needs
    // to be captured as if it were the top value of the propagationDelayBuffer.
    // When restored, it will be copied to the dest input buffer and popped off
    // before the next execution. If there is an offset, we only
    // want to capture the amount of the input buffer contributed by
    // this link.
    Array a = dest_->getData().subset(destOffset_, srcCount);
    f << a; // our part of the current Dest Input buffer.

    boost::circular_buffer<Array>::iterator itr;
    for (auto itr = srcBuffer_.begin();
         itr != srcBuffer_.end(); itr++) {
      if (itr + 1 == srcBuffer_.end())
        break; // skip the last buffer. Its the current output.
      Array &buf = *itr;
      f << buf;
    } // end for
  }
  f << "]\n";  // end of list of buffers in propagationDelayBuffer

  f << "}\n";  // end of sequence
}

void Link::deserialize(std::istream &f) {
  // Each link is a map -- extract the 9 values in the map
  // The "circularBuffer" element is a two dimentional array only present if
  // propogationDelay > 0.
  char bigbuffer[5000];
  std::string tag;
  Size count;
  std::string linkType;
  std::string linkParams;
  std::string srcRegionName;
  std::string srcOutputName;
  std::string destRegionName;
  std::string destInputName;
  Size propagationDelay;

  f >> tag;
  NTA_CHECK(tag == "{") << "Invalid network structure file -- bad link (not a map)";

  // 1. type
  f >> tag;
  NTA_CHECK(tag == "linkType:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  linkType = bigbuffer;

  // 2. params
  f >> tag;
  NTA_CHECK(tag == "params:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  linkParams = bigbuffer;

  // 3. srcRegion (name)
  f >> tag;
  NTA_CHECK(tag == "srcRegion:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  srcRegionName = bigbuffer;

  // 4. srcOutput
  f >> tag;
  NTA_CHECK(tag == "srcOutput:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  srcOutputName = bigbuffer;

  // 5. destRegion
  f >> tag;
  NTA_CHECK(tag == "destRegion:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  destRegionName = bigbuffer;

  // 6. destInput
  f >> tag;
  NTA_CHECK(tag == "destInput:");
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  destInputName = bigbuffer;

  // 7. propagationDelay (number of cycles to delay propagation)
  f >> tag;
  NTA_CHECK(tag == "propagationDelay:");
  f >> propagationDelay;

  // fill in the data for the Link object
  commonConstructorInit_(linkType, linkParams, srcRegionName, destRegionName,
                         srcOutputName, destInputName, propagationDelay);

  // 8. propagationDelayBuffer
  f >> tag;
  NTA_CHECK(tag == "propagationDelayBuffer:");
  f >> tag;
  NTA_CHECK(tag == "[")  << "Expected start of a sequence.";
  f >> count;
  // if no propagationDelay (value = 0) then there should be an empty sequence.
  NTA_CHECK(count == propagationDelay_) << "Invalid network structure file -- "
            "link has " << count << " buffers in 'propagationDelayBuffer'. "
            << "Expecting " << propagationDelay << ".";

  // Establish capacity for the requested delay data elements
  srcBuffer_.set_capacity(count);

  for (Size idx = 0; idx < count; idx++) {
    Array a;
    f >> a;
    srcBuffer_.push_back(a);
  }
  // To complete the restore, call r->prepareInputs() and then shiftBufferedData();
  // This is performed in Network class at the end of the load().
  f >> tag;
  NTA_CHECK(tag == "]");
  f >> tag;
  NTA_CHECK(tag == "}");
  f.ignore(1);
  
  if (_LINK_DEBUG)
    NTA_DEBUG << "Restored Link: " << *this;
}



bool Link::operator==(const Link &o) const {
  if (initialized_ != o.initialized_ ||
      propagationDelay_ != o.propagationDelay_ || linkType_ != o.linkType_ ||
      linkParams_ != o.linkParams_ || destOffset_ != o.destOffset_ ||
      srcRegionName_ != o.srcRegionName_ ||
      destRegionName_ != o.destRegionName_ ||
      srcOutputName_ != o.srcOutputName_ ||
      destInputName_ != o.destInputName_) {
    return false;
  }
  return true;
}


/**
 * A readable display of a Link.
 * This is not part of the save/load facility.
 */
std::ostream &operator<<(std::ostream &f, const Link &link) {
  f << "<Link>\n";
  f << "  <type>" << link.getLinkType() << "</type>\n";
  f << "  <params>" << link.getLinkParams() << "</params>\n";
  f << "  <srcRegion>" << link.getSrcRegionName() << "</srcRegion>\n";
  f << "  <destRegion>" << link.getDestRegionName() << "</destRegion>\n";
  f << "  <srcOutput>" << link.getSrcOutputName() << "</srcOutput>\n";
  f << "  <destInput>" << link.getDestInputName() << "</destInput>\n";
  f << "  <propagationDelay>" << link.getPropagationDelay()
    << "</propagationDelay>\n";
  if (link.getPropagationDelay() > 0) {
  	f <<   "  <propagationDelayBuffer>\n";
    f <<   "    <capacity> " << link.srcBuffer_.capacity() << " </capacity>\n";
    f<<    "    <size> " << link.srcBuffer_.size() << " </size>\n";
	  for (auto buf : link.srcBuffer_) {
		  f << "    " << buf << "\n";
	  }
	  f <<   "  </propagationDelayBuffer>\n";
  }
  f << "</Link>\n";
  return f;
}

} // namespace nupic
