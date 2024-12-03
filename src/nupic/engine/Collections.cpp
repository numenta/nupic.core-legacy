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

#include <nupic/engine/Spec.hpp>

/*
 * We need to import the code from Collection.cpp
 * in order to instantiate all the methods in the classes
 * instantiated below.
 */
#include <nupic/engine/Network.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/ntypes/Collection.cpp>
#include <nupic/ntypes/Collection.hpp>

using namespace nupic;

// Explicit instantiations of the collection classes used by Spec
template class nupic::Collection<OutputSpec>;
template class nupic::Collection<InputSpec>;
template class nupic::Collection<ParameterSpec>;
template class nupic::Collection<CommandSpec>;
template class nupic::Collection<Region *>;
template class nupic::Collection<Link *>;
template class nupic::Collection<Network::callbackItem>;
