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

#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>
#include <utility>

namespace nupic {
BundleIO::BundleIO(const std::string &bundlePath, const std::string &label,
                   std::string regionName, bool isInput)
    : isInput_(isInput), bundlePath_(bundlePath),
      regionName_(std::move(regionName)), ostream_(nullptr), istream_(nullptr) {
  if (!Path::exists(bundlePath_))
    NTA_THROW << "Network bundle " << bundlePath << " does not exist";

  filePrefix_ = Path::join(bundlePath, label + "-");
}

BundleIO::~BundleIO() {
  if (istream_) {
    if (istream_->is_open())
      istream_->close();
    delete istream_;
    istream_ = nullptr;
  }
  if (ostream_) {
    if (ostream_->is_open())
      ostream_->close();
    delete ostream_;
    ostream_ = nullptr;
  }
}

std::ofstream &BundleIO::getOutputStream(const std::string &name) const {
  NTA_CHECK(!isInput_);

  checkStreams_();

  ostream_ =
      new OFStream(getPath(name).c_str(), std::ios::out | std::ios::binary);
  if (!ostream_->is_open()) {
    NTA_THROW << "getOutputStream - Unable to open bundle file " << name
              << " for region " << regionName_ << " in network bundle "
              << bundlePath_;
  }

  return *ostream_;
}

std::ifstream &BundleIO::getInputStream(const std::string &name) const {
  NTA_CHECK(isInput_);

  checkStreams_();

  istream_ =
      new IFStream(getPath(name).c_str(), std::ios::in | std::ios::binary);
  if (!istream_->is_open()) {
    NTA_THROW << "getInputStream - Unable to open bundle file " << name
              << " for region " << regionName_ << " in network bundle "
              << bundlePath_;
  }

  return *istream_;
}

std::string BundleIO::getPath(const std::string &name) const {
  return filePrefix_ + name;
}

// Before a request for a new stream,
// there should be no open streams.
void BundleIO::checkStreams_() const {
  // Catch implementation errors and make it easier to
  // support direct serialization to/from archives
  if (isInput_ && istream_ != nullptr && istream_->is_open())
    NTA_THROW << "Internal Error: istream_ has not been closed";

  if (!isInput_ && ostream_ != nullptr && ostream_->is_open())
    NTA_THROW << "Internal Error: ostream_ has not been closed";
}

} // namespace nupic
