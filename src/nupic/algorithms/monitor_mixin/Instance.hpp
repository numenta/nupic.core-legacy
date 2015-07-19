/* ---------------------------------------------------------------------
* Numenta Platform for Intelligent Computing (NuPIC)
* Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
* with Numenta, Inc., for a separate license for this software code, the
* following terms and conditions apply:
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License version 3 as
* published by the Free Software Foundation.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see http://www.gnu.org/licenses.
*
* http://numenta.org/licenses/
* ----------------------------------------------------------------------
*/

/** @file
* Definitions for an abstract computational interface for modules, such as poolers
*/

#ifndef NTA_instance_HPP
#define NTA_instance_HPP

#include <vector>
#include <nupic/types/Types.hpp>

using namespace std;
using namespace nupic;

class Instance
{
public:
  string mmName;
  bool _learn;
  
  virtual void mmReset() {};

  virtual void mmCompute(vector<UInt> pattern, bool learn) { };
  virtual void mmCompute(vector<UInt> pattern, bool learn, bool learn2) { };
};

typedef void ComputeFunction(vector<UInt>& pattern, Instance& instance);

#endif // NTA_instance_HPP
