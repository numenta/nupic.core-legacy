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


#include <nupic/engine/LinkPolicy.hpp>
#include <nupic/engine/LinkPolicyFactory.hpp>
#include <nupic/engine/TestFanIn2LinkPolicy.hpp>
#include <nupic/engine/UniformLinkPolicy.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic
{


LinkPolicy* LinkPolicyFactory::createLinkPolicy(const std::string policyType, 
                                             const std::string policyParams,
                                             Link* link)
{
  LinkPolicy *lp = nullptr;
  if (policyType == "TestFanIn2")
  {
    lp = new TestFanIn2LinkPolicy(policyParams, link);
  } else if (policyType == "UniformLink")
  {
    lp = new UniformLinkPolicy(policyParams, link);
  } else if (policyType == "UnitTestLink")
  {
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
  } else if (policyType == "TestSplit")
  {
    NTA_THROW << "TestSplit not implemented yet";
  } else if (policyType == "TestOneToOne")
  {
    NTA_THROW << "TestOneToOne not implemented yet";
  } else {
    NTA_THROW << "Unknown link policy '" << policyType << "'";
  }
  return lp;
}



}

