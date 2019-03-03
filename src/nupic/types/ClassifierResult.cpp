/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <algorithm>
#include <cmath>

#include <nupic/types/ClassifierResult.hpp>
#include <nupic/utils/Log.hpp>


namespace nupic {
namespace types {

using namespace std;

ClassifierResult::~ClassifierResult() {
  for (map<Int, PDF*>::const_iterator it = result_.begin();
       it != result_.end(); ++it) {
    delete it->second;
  }
}


PDF *ClassifierResult::createVector(Int step, UInt size, Real64 value) {
  NTA_CHECK(result_.count(step) == 0) << "The ClassifierResult cannot be reused!";
  PDF *v = new PDF(size, value);
  result_.insert(pair<Int, PDF*>(step, v));
  return v;
}


bool ClassifierResult::operator==(const ClassifierResult &other) const {
  for (auto it = result_.begin(); it != result_.end(); it++) {
    const auto thisVec = it->second;
    const auto otherVec = other.result_.at(it->first);
    if (otherVec == nullptr || thisVec->size() != otherVec->size()) {
      return false;
    }
    for (UInt i = 0; i < thisVec->size(); i++) {
      if (fabs(thisVec->at(i) - otherVec->at(i)) > 0.000001) {
        return false;
      }
    }
  }
  return true;
}


UInt ClassifierResult::getClass(const UInt stepsAhead) const {
  NTA_CHECK(stepsAhead < result_.size()) << "ClassifierResult is not for steps " << stepsAhead;
  for(auto iter : this->result_) {
    if( iter.first == (Int)stepsAhead ) {  //entry at nth step  (0==current) 
      const auto *pdf = iter.second; //probability distribution of the classes
      const auto max  = std::max_element(pdf->cbegin(), pdf->cend());
      const UInt cls  = max - pdf->cbegin();
      return cls;
    }
  }
  NTA_THROW << "ClassifierResult did not match"; //should not come here
}


} // end namespace algorithms
} // end namespace nupic
