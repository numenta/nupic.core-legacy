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

#ifndef NTA_BUNDLEIO_HPP
#define NTA_BUNDLEIO_HPP

#include <iostream>
#include <nupic/os/Path.hpp>

namespace nupic {
class BundleIO {
public:
  BundleIO(std::ostream *openStream) {
    openOutputStream_ = openStream;
    openInputStream_ = nullptr;
  }
  BundleIO(std::istream *openStream) {
    openOutputStream_ = nullptr;
    openInputStream_ = openStream;
  }

  ~BundleIO() {}

  // return the stream. Caller should not close it.
  std::ostream &getOutputStream() const { return *openOutputStream_; }

  std::istream &getInputStream() const { return *openInputStream_; }

private:
  std::ostream *openOutputStream_;
  std::istream *openInputStream_;
};

} // namespace nupic

#endif // NTA_BUNDLEIO_HPP
