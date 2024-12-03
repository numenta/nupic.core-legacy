/*
 * Copyright 2013-2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <cmath>
#include <map>
#include <vector>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;

namespace nupic {
namespace algorithms {
namespace cla_classifier {

ClassifierResult::~ClassifierResult() {
  for (map<Int, vector<Real64> *>::const_iterator it = result_.begin();
       it != result_.end(); ++it) {
    delete it->second;
  }
}

vector<Real64> *ClassifierResult::createVector(Int step, UInt size,
                                               Real64 value) {
  NTA_CHECK(result_.count(step) == 0)
      << "The ClassifierResult cannot be reused!";
  vector<Real64> *v = new vector<Real64>(size, value);
  result_.insert(pair<Int, vector<Real64> *>(step, v));
  return v;
}

bool ClassifierResult::operator==(const ClassifierResult &other) const {
  for (auto it = result_.begin(); it != result_.end(); it++) {
    auto thisVec = it->second;
    auto otherVec = other.result_.at(it->first);
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

} // end namespace cla_classifier
} // end namespace algorithms
} // end namespace nupic
