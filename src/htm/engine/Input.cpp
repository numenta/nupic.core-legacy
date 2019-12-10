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
 * Implementation of Input class
 *
 */

#include <algorithm> //find

#include <htm/engine/Input.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/RegionImpl.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/ntypes/BasicType.hpp>

using namespace htm;

Input::Input(Region *region, const std::string &inputName,
             NTA_BasicType dataType)
    : region_(region), initialized_(false), data_(dataType), name_(inputName) {}

Input::~Input() {
  uninitialize();
  for (auto &link : links_) {
    std::cout << "Input::~Input: \n";
    link->getSrc()->removeLink(link); // remove it from the Output object.
    // the link is a shared_ptr so it will be deleted when links_ is cleared.
  }
  links_.clear();
}

void Input::addLink(const std::shared_ptr<Link> link, std::shared_ptr<Output> srcOutput) {
  NTA_CHECK(initialized_ == false) << "Attempt to add link to input " << name_ << " on region "
              << region_->getName() << " when input is already initialized";

  // Make sure we don't already have a link to the same output
  for (const auto &it : links_) {
    const Output* o = (*it).getSrc();
    NTA_CHECK(srcOutput.get() != o) << "Input::addLink() -- link from output="
                << srcOutput->getRegion()->getName() << "."
                << srcOutput->getName() << " to input=" << region_->getName()
                << "." << getName() << " already exists";
  }

  links_.push_back(link);

  srcOutput->addLink(link);
  // Note -- link is not usable until we set the destOffset, which
  // is calculated at initialization time
}

void Input::removeLink(const std::shared_ptr<Link> &link) {
  // removeLink should only be called internally -- if it
  // does not exist, it is a logic error
  const auto linkiter = find(links_.cbegin(), links_.cend(), link);
  NTA_CHECK(linkiter != links_.end())
      << "Cannot remove link. not found in list of links.";

  NTA_CHECK(region_->isInitialized() == false) << "Cannot remove link " << link->toString()
              << " because destination region " << region_->getName()
              << " is initialized. Remove the region first.";

  // We may have been initialized even if our containing region
  // was not. If so, uninitialize.
  uninitialize();
  link->getSrc()->removeLink(link);
  links_.erase(linkiter);
  // Link is deleted when the std::shared_ptr<Link> goes out of scope.
}

std::shared_ptr<Link> Input::findLink(const std::string &srcRegionName,
                                      const std::string &srcOutputName) {
  // Note: cannot use a map here because the link items are ordered.
  for (const auto &it: links_) {
    const Output* output = it->getSrc();
    if (output->getName() == srcOutputName &&
        output->getRegion()->getName() == srcRegionName) {
      return it;
    }
  }
  // Link not found
  return nullptr;
}

void Input::prepare() {
  // Each link copies data into its section of the overall input
  // TODO: initialization check?
  for (auto &elem : links_) {
    (elem)->compute();
  }
}


void Input::initialize() {
  /**
   * During initialization,
   *    Network calls evaluateLinks() for each region.
   *    Region calls initialize() for each input.
   *
   * Determine the Dimensions.
   * The link and region need to be consistent at both
   * ends of the link. These dimensions will be used to
   * create the Array objects for both Output and Input.
   * 1. All dimensions start out as unspecified (which means size = 0)
   * 2. If a buffer's size is set in the region Spec, this is highest priority.
   *    This would mean that the region will accept only
   *    that size input or that size output.  When creating buffers
   *    we must respect that.
   * 3. Ask dest for it's size, this is the next priority. This
   *    would be most likely from configuration.  If it returns
   *    an empty dimension then set dest dimension to 'don't care'
   *    (size=1, value[0] = 0).
   * 4. Ask source for it's size, this is the next priority. This
   *    would most likely be from configuration.  If it returns
   *    an empty dimension, set it to don't care (size=1, value[0] = 0).
   * 5. If not Fan-IN, one side is don't care, set the dimensions to the
   *    dimension of the other. If D's on both ends of the link are don't care,
   *    declare "undefined" error. If D's on both ends of the link are not equal,
   *    declare "conflict" error.
   * 6. If Fan-IN,
   *  a. consider the number of dimensions. If this is a default
   *     input on the destination region, make the source
   *     dimensions compatable with the destination region's
   *     dimensions by appending top level dimensions of 1s.
   *  b. If the contributing source dimensions have all but
   *     top dimension the same, make the destination's input
   *     dimensions the same with the top dimension being the
   *     sum of top dimensions from all contributing sources.
   *     Else, drop to 1D and make the destination dimensions
   *     to be the sum of all elements from all sources.
   *     Adjust with additional 1 dimensions to make the number
   *     of dimensions compatable as described above.
   *  c. If the destination input buffer had been originally
   *     specified, error if the new dimensions are not compatable.
   */
  if (initialized_)
    return;


  const std::shared_ptr<Spec> &destSpec = region_->getSpec();
  bool regionLevel = destSpec->inputs.getByName(name_).regionLevel;
  UInt32 total_width = 0u;
  size_t maxD = 1;
  Dimensions d;
  Dimensions inD = dim_;
  bool is_FanIn = links_.size() > 1;

  // First determine the original configuration for destination dimensions.
  // Most of the time we get our input dimensions from a connected output
  // but here we want to see if there was an override.
  // Normally this will be 'don't care' if there is no override.
  if (!inD.isSpecified()) {
    // ask the spec for destination region.
    UInt32 count = destSpec->inputs.getByName(name_).count;
    if (count > 0) {
      inD.push_back(count);
    } else {
      // ask the destination region impl
      inD = region_->askImplForInputDimensions(name_);
      if (inD.empty()) {
        inD.push_back(0); // set Don't care.
      }
    }
  }

  if (links_.size() > 0) {
    // We have links.
    // Try to determine source dimensions.
    std::vector<Dimensions> Ds;
    for (auto link : links_) {
      Output* out = link->getSrc();
      // determines source dimensions based on configuration.
      d = out->determineDimensions();
      NTA_CHECK(!d.isUnspecified());
      if (d.isDontcare()) {
        d = inD; // use destination dimensions for source. rare.
        out->setDimensions(d);
      }
      NTA_CHECK(d.isSpecified())
          << "Cannot determine the dimensions for Output " << out->getRegion()->getName()
          << "." << out->getName();

      out->initialize(); // creates the output buffers.

      // Initialize Link.  'total_width' at this point is the byte offset
      // into the input buffer where the output will start writing.
      link->initialize(total_width, is_FanIn);
      total_width += (UInt32)d.getCount();

      if (is_FanIn) {
        // save some info, we will need it later.
        Ds.push_back(d);
        size_t n = d.size();
        while (n > maxD && d[n - 1] == 1)
          n--; // ignore top level dimensions of 1
        if (n > maxD)
          maxD = n;
      } else {
        // Not a FanIn.
        if (inD.isSpecified()) {
          NTA_CHECK(inD.getCount() == d.getCount())
              << "Dimensions were specified for input "
              << region_->getName() << "." << name_ << " " << inD
              << " but it is inconsistant with the dimensions of the source "
              << out->getRegion()->getName() << "." << out->getName() << " " << d;
          // else keep the manually configured dimensions.
        } else {
          inD = d;  // set the destination dimensions to be same as source.
        }
      }
    }

    if (is_FanIn) {
      // Try to figure out the destination dimensions derived
      // from the source dimensions. If any source dimension other
      // than the top level did not match we will have to flatten
      // everything and use 1D.
      // example:
      //   sources
      //       [100, 10]
      //       [100, 5 ]
      //       [100]
      //   Fan in to:
      //       [100, 16]
      // We also add up the top level while we are doing it.
      bool match = true;
      UInt32 topsum = 0;
      for (size_t i = 0; match && i < maxD - 1; i++) {
        topsum = 0;
        for (size_t j = 0; match && j < Ds.size(); j++) {
          if ((i + 1) >= Ds[j].size()) {
            Ds[j].push_back(1);
          } // fill with 1's if needed
          if (Ds[0][i] != Ds[j][i])
            match = false; // This dimension did not match.
          topsum += Ds[j][i + 1];
        }
      }
      // at this point:
      //     if match = false, we have to flatten to 1 X total_width.
      //     if not, topsum is the top dimension for our destination
      //     maxD-1 is its index. The lower dimensions can be taken from any
      //     source.
      d.clear();
      if (match && topsum > 0) {
        for (size_t i = 0; i < maxD - 1; i++)
          d.push_back(Ds[0][i]);
        d.push_back(topsum);
      } else {
        d.push_back(total_width);
      }
      if (inD.isSpecified()) {
        NTA_CHECK(inD.getCount() == d.getCount())
            << "Dimensions were specified for " << region_->getName()
            << "." << name_
            << " but it is inconsistant with the dimensions of the sources.";
        // keep the manually configured dimensions.
      } else {
        inD = d;
      }
    }
  } // end of link iteration.

  // If this is the regionLevel input and the region dim is don't care,
  // then assign this input dimensions to the region dimensions.
  // The region dimensions must have the same number of dimensions.
  // Add 1's as needed to either.
  if (regionLevel && inD.isSpecified()) {
    d = region_->getDimensions();
    if (d.isSpecified()) {
      maxD = d.size();
      while (inD.size() < maxD)
        inD.push_back(1);
      while (d.size() < inD.size())
        d.push_back(1);
    } else {
      d = inD;
    }
    if (!region_->isInitialized())
      region_->setDimensions(d);
  }

  if (links_.size() > 0) {
    NTA_CHECK(inD.isSpecified()) << "Input " << region_->getName() << "." << name_
                      << " has an incoming link but no dimensions are configured.";
  }
  // Create the Input buffer.
  dim_ = inD;

  if (data_.getType() == NTA_BasicType_SDR) {
    data_.allocateBuffer(dim_.asVector());
  } else if (dim_.isDontcare()) {
    data_.allocateBuffer(0);  // lets hope this is an unused input.
  } else {
    data_.allocateBuffer(dim_.getCount());
    data_.zeroBuffer();
  }

  initialized_ = true;
}

void Input::uninitialize() {
  if (!initialized_)
    return;

  NTA_CHECK(!region_->isInitialized());

  initialized_ = false;
  data_.releaseBuffer();
}

// This is called when an output is resized.
// We need to change the size of the dimensions and the input buffer.
// If this is a Fan-in then we need to adjust the offset on the links.
void Input::resize() {
  if (!initialized_) return;
  size_t count = 0;
  for (auto link : links_) {
    Output *out = link->getSrc();
    link->setOffset(count);
    count += out->getData().getCount();
  }
  dim_ = {static_cast<UInt32>(count)};
  data_.allocateBuffer(count);
}

namespace htm {
  std::ostream &operator<<(std::ostream &f, const Input &d) {
    f << "Input:  " << d.getRegion()->getName() << "." << d.getName() << " dim:" << d.dim_ << " buffer:" << d.getData();
    return f;
  }
}

bool Input::isInitialized() { return (initialized_); }

void Input::setName(const std::string &name) { name_ = name; }

const std::string &Input::getName() const { return name_; }

