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

/** @file
 * Definition of the LinkPolicyFactory API
 */

#ifndef NTA_LINKPOLICY_FACTORY_HPP
#define NTA_LINKPOLICY_FACTORY_HPP

#include <string>

namespace nupic {

class LinkPolicy;
class Link;
class Region;

class LinkPolicyFactory {
public:
  // LinkPolicyFactory is a lightweight object
  LinkPolicyFactory(){};
  ~LinkPolicyFactory(){};

  // Create a LinkPolicy of a specific type; caller gets ownership.
  LinkPolicy *createLinkPolicy(const std::string policyType,
                               const std::string policyParams, Link *link);

private:
};

} // namespace nupic

#endif // NTA_LINKPOLICY_FACTORY_HPP
