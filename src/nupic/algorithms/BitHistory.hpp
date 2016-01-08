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

/** @file
 * Definitions for the BitHistory.
 */

#ifndef NTA_bit_history_HPP
#define NTA_bit_history_HPP

#include <map>
#include <string>
#include <vector>

#include <nupic/proto/BitHistory.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>

using namespace std;

namespace nupic
{
  namespace algorithms
  {
    namespace cla_classifier
    {

      /** Class to store duty cycles for buckets for a single input bit.
       *
       * @b Responsibility
       * The BitHistory is responsible for updating and relaying the duty
       * cycles for the different buckets.
       *
       * TODO: Support serialization and deserialization.
       *
       */
      class BitHistory : public Serializable<BitHistoryProto>
      {
        public:
          /**
           * Constructor.
           */
          BitHistory() {}

          /**
           * Constructor.
           *
           * @param bitNum The input bit index that this BitHistory stores data
           *               for.
           * @param nSteps The number of steps this BitHistory is storing duty
           *               cycles for.
           * @param alpha The alpha to use when decaying the duty cycles.
           * @param verbosity The logging verbosity to use.
           *
           */
          BitHistory(UInt bitNum, int nSteps, Real64 alpha, UInt verbosity);

          virtual ~BitHistory() {};

          /**
           * Update the duty cycle for the specified bucket index.
           *
           * @param iteration The current iteration. The difference between
           *                  consecutive calls is used to determine how much to
           *                  decay the previous duty cycle value.
           * @param bucketIdx The bucket index to update.
           *
           */
          void store(int iteration, int bucketIdx);

          /**
           * Sets the votes for each bucket when this cell is active.
           *
           * @param iteration The current iteration.
           * @param votes A vector to populate with the votes for each bucket.
           *
           */
          void infer(int iteration, vector<Real64>* votes);

          /**
           * Save the state to the ostream.
           */
          void save(ostream& outStream) const;

          /**
           * Load state from istream.
           */
          void load(istream& inStream);

          /**
           * Save the state to the builder.
           */
          using Serializable::write;
          void write(BitHistoryProto::Builder& builder) const;

          /**
           * Load state from reader.
           */
          using Serializable::read;
          void read(BitHistoryProto::Reader& proto);

          /**
           * Check if the other instance matches this one.
           *
           * @param other an instance to compare to
           * @returns true iff the other instance matches this one
           */
          bool operator==(const BitHistory& other) const;

          /**
           * Check if the other instance doesn't match this one.
           *
           * @param other an instance to compare to
           * @returns true iff the other instance matches doesn't match this one
           */
          bool operator!=(const BitHistory& other) const;

        private:

          string id_;
          // Mapping from bucket index to the duty cycle values.
          map<int, Real64> stats_;
          // Last iteration at which the duty cycles were updated to the present
          // value. This is not done every iteration for efficiency reasons.
          int lastTotalUpdate_;
          int learnIteration_;
          // The alpha to use when decaying the duty cycles.
          Real64 alpha_;
          UInt verbosity_;
      }; // end class BitHistory

    } // end namespace cla_classifier
  } // end namespace algorithms
} // end namespace nupic

#endif // NTA_fast_cla_classifier_HPP
