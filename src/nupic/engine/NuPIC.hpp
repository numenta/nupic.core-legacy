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

/**
 * @file
 * NuPIC init/shutdown operations
 */

#include <set>

/** @namespace nupic
 *
 * Contains the primary NuPIC API.
 */
namespace nupic {
class Network;

/**
 * Initialization and shutdown operations for NuPIC engine.
 */
class NuPIC {
public:
  /**
   * Initialize NuPIC.
   *
   * @note It's safe to reinitialize an initialized NuPIC.
   * @note Creating a Network will auto-initialize NuPIC.
   */
  static void init();

  /**
   * Shutdown NuPIC.
   *
   * @note As a safety measure, NuPIC with any Network still registered to it
   * is not allowed to be shut down.
   */
  static void shutdown();

  /**
   *
   * @return Whether NuPIC is initialized successfully.
   */
  static bool isInitialized();

private:
  /**
   * Having Network as friend class to allow Networks register/unregister
   * themselves at creation and destruction time by calling non-public methods
   * of NuPIC.
   *
   */
  friend class Network;

  static void registerNetwork(Network *net);
  static void unregisterNetwork(Network *net);
  static std::set<Network *> networks_;
  static bool initialized_;
};
} // namespace nupic
