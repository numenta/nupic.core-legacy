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

/* @file Implementation of NuPIC init/shutdown operations */

// TODO -- thread safety

#include <apr-1/apr_general.h>
#include <nupic/engine/NuPIC.hpp>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

std::set<Network *> NuPIC::networks_;
bool NuPIC::initialized_ = false;

void NuPIC::init() {
  if (isInitialized())
    return;

  // internal consistency check. Nonzero should be impossible.
  NTA_CHECK(networks_.size() == 0) << "Internal error in NuPIC::init()";

  // Initialize APR as a library client
  // TODO: move to OS::initialize()?
  int result = apr_initialize();
  if (result)
    NTA_THROW << "Error initializing APR (code " << result << ")";

  initialized_ = true;
}

void NuPIC::shutdown() {
  if (!isInitialized()) {
    NTA_THROW << "NuPIC::shutdown -- NuPIC has not been initialized";
  }

  if (!networks_.empty()) {
    NTA_THROW << "NuPIC::shutdown -- cannot shut down NuPIC because "
              << networks_.size() << " networks still exist.";
  }

  RegionImplFactory::getInstance().cleanup();
  initialized_ = false;
}

void NuPIC::registerNetwork(Network *net) {
  if (!isInitialized()) {
    NTA_THROW
        << "Attempt to create a network before NuPIC has been initialized -- "
           "call NuPIC::init() before creating any networks";
  }

  auto n = networks_.find(net);
  // This should not be possible
  NTA_CHECK(n == networks_.end())
      << "Internal error -- double registration of network";
  networks_.insert(net);
}

void NuPIC::unregisterNetwork(Network *net) {
  auto n = networks_.find(net);
  NTA_CHECK(n != networks_.end()) << "Internal error -- network not registered";
  networks_.erase(n);
}

bool NuPIC::isInitialized() { return initialized_; }

} // namespace nupic
