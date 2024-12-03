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

#include <nupic/engine/LinkPolicy.hpp>
#include <nupic/engine/LinkPolicyFactory.hpp>
#include <nupic/engine/TestFanIn2LinkPolicy.hpp>
#include <nupic/engine/UniformLinkPolicy.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

LinkPolicy *LinkPolicyFactory::createLinkPolicy(const std::string policyType,
                                                const std::string policyParams,
                                                Link *link) {
  LinkPolicy *lp = nullptr;
  if (policyType == "TestFanIn2") {
    lp = new TestFanIn2LinkPolicy(policyParams, link);
  } else if (policyType == "UniformLink") {
    lp = new UniformLinkPolicy(policyParams, link);
  } else if (policyType == "UnitTestLink") {
    // When unit testing a link policy, a valid Link* is required to be passed
    // to the link policy's constructor.  If you pass NULL, other portions of
    // NuPIC may try to dereference it (e.g. operator<< from NTA_THROW).  So we
    // allow for a UnitTestLink link policy which doesn't actually provide
    // anything.  This way, you can create a dummy link like so:
    //
    // Link dummyLink("UnitTestLink", "", "", "");
    //
    // and pass this dummy link to the constructor of the real link policy
    // you wish to unit test.
  } else if (policyType == "TestSplit") {
    NTA_THROW << "TestSplit not implemented yet";
  } else if (policyType == "TestOneToOne") {
    NTA_THROW << "TestOneToOne not implemented yet";
  } else {
    NTA_THROW << "Unknown link policy '" << policyType << "'";
  }
  return lp;
}

} // namespace nupic
