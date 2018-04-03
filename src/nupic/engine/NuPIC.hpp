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

/**
 * @file
 * NuPIC init/shutdown operations
 */

#include <set>

/** @namespace nupic
 *
 * Contains the primary NuPIC API.
 */
namespace nupic
{
  class Network;

  /**
   * Initialization and shutdown operations for NuPIC engine.
   */
  class NuPIC 
  {
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
     * themselves at creation and destruction time by calling non-public methods of NuPIC.
     *
     */    
    friend class Network;

    static void registerNetwork(Network* net);
    static void unregisterNetwork(Network* net);
    static std::set<Network*> networks_;
    static bool initialized_;
  };
} // namespace nupic

