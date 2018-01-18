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


#include <nupic/engine/Spec.hpp>

/*
 * We need to import the code from Collection.cpp 
 * in order to instantiate all the methods in the classes
 * instantiated below. 
 */
#include <nupic/ntypes/Collection.hpp>
#include <nupic/ntypes/Collection.cpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Network.hpp>

using namespace nupic;


// Explicit instantiations of the collection classes used by Spec
template class nupic::Collection<OutputSpec>;
template class nupic::Collection<InputSpec>;
template class nupic::Collection<ParameterSpec>;
template class nupic::Collection<CommandSpec>;
template class nupic::Collection<Region*>;
template class nupic::Collection<Link*>;
template class nupic::Collection<Network::callbackItem>;
