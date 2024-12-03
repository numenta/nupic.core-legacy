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

#ifndef NTA_classifier_result_HPP
#define NTA_classifier_result_HPP

#include <map>
#include <vector>

#include <nupic/types/Types.hpp>

using namespace std;

namespace nupic {
namespace algorithms {
namespace cla_classifier {

/** CLA classifier result class.
 *
 * @b Responsibility
 * The ClassifierResult is responsible for storing result data and
 * cleaning up the data when deleted.
 *
 */
class ClassifierResult {
public:
  /**
   * Constructor.
   */
  ClassifierResult() {}

  /**
   * Destructor - frees memory allocated during lifespan.
   */
  virtual ~ClassifierResult();

  /**
   * Creates and returns a vector for a given step.
   *
   * The vectors created are stored and can be accessed with the
   * iterator methods. The vectors are owned by this class and are
   * deleted in the destructor.
   *
   * @param step The prediction step to create a vector for. If -1, then
   *             a vector for the actual values to use for each bucket
   *             is returned.
   * @param size The size of the desired vector.
   * @param value The value to populate the vector with.
   *
   * @returns The specified vector.
   */
  virtual vector<Real64> *createVector(Int step, UInt size, Real64 value);

  /**
   * Checks if the other instance has the exact same values.
   *
   * @param other The other instance to compare to.
   * @returns True iff the other instance has the same values.
   */
  virtual bool operator==(const ClassifierResult &other) const;

  /**
   * Iterator method begin.
   */
  virtual map<Int, vector<Real64> *>::const_iterator begin() {
    return result_.begin();
  }

  /**
   * Iterator method end.
   */
  virtual map<Int, vector<Real64> *>::const_iterator end() {
    return result_.end();
  }

private:
  map<Int, vector<Real64> *> result_;

}; // end class ClassifierResult

} // end namespace cla_classifier
} // end namespace algorithms
} // end namespace nupic

#endif // NTA_classifier_result_HPP
