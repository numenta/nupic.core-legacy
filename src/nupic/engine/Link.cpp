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
#include <nupic/engine/Link.hpp>
#include <nupic/utils/ArrayProtoUtils.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/engine/LinkPolicyFactory.hpp>
#include <nupic/engine/LinkPolicy.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/BasicType.hpp>

namespace nupic
{

Link::Link(const std::string& linkType, const std::string& linkParams,
           const std::string& srcRegionName, const std::string& destRegionName,
           const std::string& srcOutputName, const std::string& destInputName,
           const size_t propagationDelay):
             srcBuffer_(0)
{
  commonConstructorInit_(linkType, linkParams,
        srcRegionName, destRegionName,
        srcOutputName, destInputName,
        propagationDelay);

}

Link::Link(const std::string& linkType, const std::string& linkParams,
           Output* srcOutput, Input* destInput, const size_t propagationDelay):
             srcBuffer_(0)
{
  commonConstructorInit_(linkType, linkParams,
        srcOutput->getRegion().getName(),
        destInput->getRegion().getName(),
        srcOutput->getName(),
        destInput->getName(),
        propagationDelay);

  connectToNetwork(srcOutput, destInput);
  // Note -- link is not usable until we set the destOffset, which happens at initialization time
}


Link::Link():
            srcBuffer_(0)
{
}


void Link::commonConstructorInit_(const std::string& linkType, const std::string& linkParams,
                 const std::string& srcRegionName, const std::string& destRegionName,
                 const std::string& srcOutputName,  const std::string& destInputName,
                 const size_t propagationDelay)
{
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

Link::~Link()
{
  delete impl_;
}


void Link::initPropagationDelayBuffer_(size_t propagationDelay,
                                       NTA_BasicType dataElementType,
                                       size_t dataElementCount)
{
  if (srcBuffer_.capacity() != 0)
  {
    // Already initialized; e.g., as result of de-serialization
    return;
  }

  // Establish capacity for the requested delay data elements plus one slot for
  // the next output element
  srcBuffer_.set_capacity(propagationDelay + 1);

  // Initialize delay data elements
  size_t dataBufferSize = dataElementCount *
                          BasicType::getSize(dataElementType);

  for(size_t i=0; i < propagationDelay; i++)
  {
    Array arrayTemplate(dataElementType);

    srcBuffer_.push_back(arrayTemplate);

    // Allocate 0-initialized data for current element
    srcBuffer_[i].allocateBuffer(dataElementCount);
    ::memset(srcBuffer_[i].getBuffer(), 0, dataBufferSize);
  }
}


void Link::initialize(size_t destinationOffset)
{
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
  const Dimensions& srcD = getSrcDimensions();
  const Dimensions& destD = getDestDimensions();
  NTA_CHECK(! srcD.isUnspecified());
  NTA_CHECK(! destD.isUnspecified());

  Dimensions oneD;
  oneD.push_back(1);

  if(src_->isRegionLevel())
  {
    Dimensions d;
    for(size_t i = 0; i < src_->getRegion().getDimensions().size(); i++)
    {
      d.push_back(1);
    }

    NTA_CHECK(srcD.isDontcare() || srcD == d);
  }
  else if(src_->getRegion().getDimensions() == oneD)
  {
    Dimensions d;
    for(size_t i = 0; i < srcD.size(); i++)
    {
      d.push_back(1);
    }
    NTA_CHECK(srcD.isDontcare() || srcD == d);
  }
  else
  {
    NTA_CHECK(srcD.isDontcare() || srcD == src_->getRegion().getDimensions());
  }

  if(dest_->isRegionLevel())
  {
    Dimensions d;
    for(size_t i = 0; i < dest_->getRegion().getDimensions().size(); i++)
    {
      d.push_back(1);
    }

    NTA_CHECK(destD.isDontcare() || destD.isOnes());
  }
  else if(dest_->getRegion().getDimensions() == oneD)
  {
    Dimensions d;
    for(size_t i = 0; i < destD.size(); i++)
    {
      d.push_back(1);
    }
    NTA_CHECK(destD.isDontcare() || destD == d);
  }
  else
  {
    NTA_CHECK(destD.isDontcare() || destD == dest_->getRegion().getDimensions());
  }

  destOffset_ = destinationOffset;
  impl_->initialize();

  // ---
  // Initialize the propagation delay buffer
  // ---
  initPropagationDelayBuffer_(propagationDelay_,
                              src_->getData().getType(),
                              src_->getData().getCount());

  initialized_ = true;

}

void Link::setSrcDimensions(Dimensions& dims)
{
  NTA_CHECK(src_ != nullptr && dest_ != nullptr)
    << "Link::setSrcDimensions() can only be called on a connected link";

  size_t nodeElementCount = src_->getNodeOutputElementCount();
  if(nodeElementCount == 0)
  {
    nodeElementCount =
      src_->getRegion().getNodeOutputElementCount(src_->getName());
  }
  impl_->setNodeOutputElementCount(nodeElementCount);

  impl_->setSrcDimensions(dims);
}

void Link::setDestDimensions(Dimensions& dims)
{
  NTA_CHECK(src_ != nullptr && dest_ != nullptr)
    << "Link::setDestDimensions() can only be called on a connected link";

  size_t nodeElementCount = src_->getNodeOutputElementCount();
  if(nodeElementCount == 0)
  {
    nodeElementCount =
      src_->getRegion().getNodeOutputElementCount(src_->getName());
  }
  impl_->setNodeOutputElementCount(nodeElementCount);

  impl_->setDestDimensions(dims);
}

const Dimensions& Link::getSrcDimensions() const
{
  return impl_->getSrcDimensions();
};

const Dimensions& Link::getDestDimensions() const
{
  return impl_->getDestDimensions();
};

// Return constructor params
const std::string& Link::getLinkType() const
{
  return linkType_;
}

const std::string& Link::getLinkParams() const
{
  return linkParams_;
}

const std::string& Link::getSrcRegionName() const
{
  return srcRegionName_;
}

const std::string& Link::getSrcOutputName() const
{
  return srcOutputName_;
}

const std::string& Link::getDestRegionName() const
{
  return destRegionName_;
}

const std::string& Link::getDestInputName() const
{
  return destInputName_;
}

const std::string Link::toString() const
{
  std::stringstream ss;
  ss << "[" << getSrcRegionName() << "." << getSrcOutputName();
  if (src_)
  {
    ss << " (region dims: " << src_->getRegion().getDimensions().toString() << ") ";
  }
  ss << " to " << getDestRegionName() << "." << getDestInputName() ;
  if (dest_)
  {
    ss << " (region dims: " << dest_->getRegion().getDimensions().toString() << ") ";
  }
  ss << " type: " << linkType_ << "]";
  return ss.str();
}

void Link::connectToNetwork(Output *src, Input *dest)
{
  NTA_CHECK(src != nullptr);
  NTA_CHECK(dest != nullptr);

  src_ = src;
  dest_ = dest;
}


// The methods below only work on connected links.
Output& Link::getSrc() const

{
  NTA_CHECK(src_ != nullptr)
    << "Link::getSrc() can only be called on a connected link";
  return *src_;
}

Input& Link::getDest() const
{
  NTA_CHECK(dest_ != nullptr)
    << "Link::getDest() can only be called on a connected link";
  return *dest_;
}

void
Link::buildSplitterMap(Input::SplitterMap& splitter)
{
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

  for (size_t destNode = 0; destNode < splitter.size(); destNode++)
  {
    // convert proto-splitter values into real
    // splitter values;
    for (auto & elem : protoSplitter[destNode])
    {
      size_t srcElement = elem;
      size_t elementOffset = srcElement + destOffset_;
      splitter[destNode].push_back(elementOffset);
    }

  }
}

void
Link::compute()
{
  NTA_CHECK(initialized_);

  // If first compute during current network run iteration, append src to
  // circular buffer
  if (!srcBuffer_.full())
  {
    const Array & srcArray = src_->getData();
    size_t elementCount = srcArray.getCount();
    auto elementType = srcArray.getType();

    Array array(elementType);
    srcBuffer_.push_back(array);

    auto & lastElement = srcBuffer_.back();
    lastElement.allocateBuffer(elementCount);
    ::memcpy(lastElement.getBuffer(), srcArray.getBuffer(),
             elementCount * BasicType::getSize(elementType));
  }

  // Copy data from source to destination.
  const Array & src = srcBuffer_[0];
  const Array & dest = dest_->getData();

  size_t typeSize = BasicType::getSize(src.getType());
  size_t srcSize = src.getCount() * typeSize;
  size_t destByteOffset = destOffset_ * typeSize;
  ::memcpy((char*)(dest.getBuffer()) + destByteOffset, src.getBuffer(), srcSize);
}

void Link::purgeBufferHead()
{
  NTA_CHECK(!srcBuffer_.empty());

  srcBuffer_.pop_front();
}

void Link::write(LinkProto::Builder& proto) const
{
  proto.setType(linkType_.c_str());
  proto.setParams(linkParams_.c_str());
  proto.setSrcRegion(srcRegionName_.c_str());
  proto.setSrcOutput(srcOutputName_.c_str());
  proto.setDestRegion(destRegionName_.c_str());
  proto.setDestInput(destInputName_.c_str());

  // Save delayed outputs
  auto delayedOutputsBuilder = proto.initDelayedOutputs(propagationDelay_);
  for (size_t i=0; i < propagationDelay_; ++i)
  {
    ArrayProtoUtils::copyArrayToArrayProto(srcBuffer_[i],
                                           delayedOutputsBuilder[i]);
  }
}


void Link::read(LinkProto::Reader& proto)
{
  const auto delayedOutputsReader = proto.getDelayedOutputs();

  commonConstructorInit_(
      proto.getType().cStr(), proto.getParams().cStr(),
      proto.getSrcRegion().cStr(), proto.getDestRegion().cStr(),
      proto.getSrcOutput().cStr(), proto.getDestInput().cStr(),
      delayedOutputsReader.size()/*propagationDelay*/);

  if (delayedOutputsReader.size())
  {
    // Initialize the propagation delay buffer with delay array buffers having 0
    // elements that deserialization logic will replace with appropriately-sized
    // buffers.
    initPropagationDelayBuffer_(
      propagationDelay_,
      ArrayProtoUtils::getArrayTypeFromArrayProtoReader(delayedOutputsReader[0]),
      0);

    // Populate delayed outputs

    for (size_t i=0; i < propagationDelay_; ++i)
    {
      ArrayProtoUtils::copyArrayProtoToArray(delayedOutputsReader[i],
                                             srcBuffer_[i],
                                             true/*allocArrayBuffer*/);
    }
  }
}


namespace nupic
{
  std::ostream& operator<<(std::ostream& f, const Link& link)
  {
    f << "<Link>\n";
    f << "  <type>" << link.getLinkType() << "</type>\n";
    f << "  <params>" << link.getLinkParams() << "</params>\n";
    f << "  <srcRegion>" << link.getSrcRegionName() << "</srcRegion>\n";
    f << "  <destRegion>" << link.getDestRegionName() << "</destRegion>\n";
    f << "  <srcOutput>" << link.getSrcOutputName() << "</srcOutput>\n";
    f << "  <destInput>" << link.getDestInputName() << "</destInput>\n";
    f << "</Link>\n";
    return f;
  }
}

}
