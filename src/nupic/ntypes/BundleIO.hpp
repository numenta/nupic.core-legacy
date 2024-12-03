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

#ifndef NTA_BUNDLEIO_HPP
#define NTA_BUNDLEIO_HPP

#include <nupic/os/FStream.hpp>
#include <nupic/os/Path.hpp>

namespace nupic {
class BundleIO {
public:
  BundleIO(const std::string &bundlePath, const std::string &label,
           std::string regionName, bool isInput);

  ~BundleIO();

  // These are {o,i}fstream instead of {o,i}stream so that
  // the node can explicitly close() them.
  std::ofstream &getOutputStream(const std::string &name) const;

  std::ifstream &getInputStream(const std::string &name) const;

  std::string getPath(const std::string &name) const;

private:
  // Before a request for a new stream,
  // there should be no open streams.
  void checkStreams_() const;

  // Should never read and write at the same time -- this helps
  // to enforce.
  bool isInput_;

  // We only need the file prefix, but store the bundle path
  // for error messages
  std::string bundlePath_;

  // Store the whole prefix instead of just the label
  std::string filePrefix_;

  // Store the region name for debugging
  std::string regionName_;

  // We own the streams -- helps with finding errors
  // and with enforcing one-stream-at-a-time
  // These are mutable because the bundle doesn't conceptually
  // change when you serialize/deserialize.
  mutable std::ofstream *ostream_;
  mutable std::ifstream *istream_;

}; // class BundleIO
} // namespace nupic

#endif // NTA_BUNDLEIO_HPP
