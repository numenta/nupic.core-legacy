/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

#ifndef NTA_GROUPBY_HPP
#define NTA_GROUPBY_HPP

#include <algorithm> // is_sorted
#include <tuple>

#include <htm/utils/Log.hpp>

namespace htm {
/** @file
 * Implements a groupBy function.
 *
 * This is modeled after Python's itertools.groupby, but with the added
 * ability to traverse multiple sequences. Similar to Python, it requires the
 * input to be sorted according to the supplied key functions.
 *
 * There are two functions:
 *
 * - `groupBy`, which takes in collections
 * - `iterGroupBy`, which takes in pairs of iterators
 *
 * Both functions take a key function for each sequence.
 *
 * Both functions return an iterable object. The iterator returns a tuple
 * containing the key, followed by a begin and end iterator for each
 * sequence. The sequences are traversed lazily as the iterator is advanced.
 *
 * Note: The implementation includes this "minFrontKey" to avoid GCC
 * "maybe-initialized" false positives. This approach makes it very obvious to
 * the compiler that the "key" variable always gets initialized.
 *
 * Feel free to add new GroupBy7, GroupBy8, ..., GroupByN classes as needed.
 */

// ==========================================================================
// 3 SEQUENCES
// ==========================================================================
// groupBy used only in TM.cpp
//  //TODO see if could be simplified and moved to TM as a private method groupBy_ 

template <typename Iterator0, typename KeyFn0,
          typename KeyType = typename std::remove_const<typename std::result_of<
              KeyFn0(decltype(*std::declval<Iterator0>()))>::type>::type>
static KeyType minFrontKey(KeyType frontrunner, Iterator0 begin0,
                           Iterator0 end0, KeyFn0 keyFn0) {
  if (begin0 != end0) {
    return std::min(frontrunner, keyFn0(*begin0));
  } else {
    return frontrunner;
  }
}

template <typename Iterator0, typename KeyFn0, typename Iterator1,
          typename KeyFn1,
          typename KeyType = typename std::remove_const<typename std::result_of<
              KeyFn0(decltype(*std::declval<Iterator0>()))>::type>::type>
static KeyType minFrontKey(KeyType frontrunner, Iterator0 begin0,
                           Iterator0 end0, KeyFn0 keyFn0, Iterator1 begin1,
                           Iterator1 end1, KeyFn1 keyFn1) {
  KeyType ret = frontrunner;

  if (begin0 != end0) {
    ret = std::min(ret, static_cast<KeyType>(keyFn0(*begin0)));
  }

  if (begin1 != end1) {
    ret = std::min(ret, keyFn1(*begin1));
  }

  return ret;
}

template <typename Iterator0, typename KeyFn0, typename Iterator1,
          typename KeyFn1, typename Iterator2, typename KeyFn2,
          typename Element0 = decltype(*std::declval<Iterator0>()),
          typename Element1 = decltype(*std::declval<Iterator1>()),
          typename Element2 = decltype(*std::declval<Iterator2>()),
          typename KeyType = typename std::remove_const<typename std::result_of<
              KeyFn0(decltype(*std::declval<Iterator0>()))>::type>::type>
class GroupBy3 {
public:
  GroupBy3(Iterator0 begin0, Iterator0 end0, KeyFn0 keyFn0, Iterator1 begin1,
           Iterator1 end1, KeyFn1 keyFn1, Iterator2 begin2, Iterator2 end2,
           KeyFn2 keyFn2)
      : begin0_(begin0), end0_(end0), keyFn0_(keyFn0), begin1_(begin1),
        end1_(end1), keyFn1_(keyFn1), begin2_(begin2), end2_(end2),
        keyFn2_(keyFn2) {
    NTA_ASSERT(
        std::is_sorted(begin0, end0, [&](const Element0 &a, const Element0 &b) {
          return keyFn0(a) < keyFn0(b);
        }));
    NTA_ASSERT(
        std::is_sorted(begin1, end1, [&](const Element1 &a, const Element1 &b) {
          return keyFn1(a) < keyFn1(b);
        }));
    NTA_ASSERT(
        std::is_sorted(begin2, end2, [&](const Element2 &a, const Element2 &b) {
          return keyFn2(a) < keyFn2(b);
        }));
  }

  class Iterator {
  public:
    Iterator(Iterator0 begin0, Iterator0 end0, KeyFn0 keyFn0, Iterator1 begin1,
             Iterator1 end1, KeyFn1 keyFn1, Iterator2 begin2, Iterator2 end2,
             KeyFn2 keyFn2)
        : current0_(begin0), end0_(end0), keyFn0_(keyFn0), current1_(begin1),
          end1_(end1), keyFn1_(keyFn1), current2_(begin2), end2_(end2),
          keyFn2_(keyFn2), finished_(false) {
      calculateNext_();
    }

    bool operator!=(const Iterator &other) {
      return (finished_ != other.finished_ || current0_ != other.current0_ ||
              current1_ != other.current1_ || current2_ != other.current2_);
    }

    const std::tuple<KeyType, Iterator0, Iterator0, Iterator1, Iterator1,
                     Iterator2, Iterator2> &
    operator*() const {
      NTA_ASSERT(!finished_);
      return v_;
    }

    const Iterator &operator++() {
      NTA_ASSERT(!finished_);
      calculateNext_();
      return *this;
    }

  private:
    void calculateNext_() {
      if (current0_ != end0_ || current1_ != end1_ || current2_ != end2_) {
        // Find the lowest key.
        KeyType key;
        if (current0_ != end0_) {
          key = minFrontKey(keyFn0_(*current0_), current1_, end1_, keyFn1_,
                            current2_, end2_, keyFn2_);
        } else if (current1_ != end1_) {
          key = minFrontKey(keyFn1_(*current1_), current2_, end2_, keyFn2_);
        } else {
          key = keyFn2_(*current2_);
        }

        std::get<0>(v_) = key;

        // Find all elements with this key.
        std::get<1>(v_) = current0_;
        while (current0_ != end0_ && keyFn0_(*current0_) == key) {
          current0_++;
        }
        std::get<2>(v_) = current0_;

        std::get<3>(v_) = current1_;
        while (current1_ != end1_ && keyFn1_(*current1_) == key) {
          current1_++;
        }
        std::get<4>(v_) = current1_;

        std::get<5>(v_) = current2_;
        while (current2_ != end2_ && keyFn2_(*current2_) == key) {
          current2_++;
        }
        std::get<6>(v_) = current2_;
      } else {
        finished_ = true;
      }
    }

    std::tuple<KeyType, Iterator0, Iterator0, Iterator1, Iterator1, Iterator2, Iterator2> v_;

    Iterator0 current0_;
    Iterator0 end0_;
    KeyFn0 keyFn0_;

    Iterator1 current1_;
    Iterator1 end1_;
    KeyFn1 keyFn1_;

    Iterator2 current2_;
    Iterator2 end2_;
    KeyFn2 keyFn2_;

    bool finished_;
  };

  Iterator begin() const {
    return Iterator(begin0_, end0_, keyFn0_, begin1_, end1_, keyFn1_, begin2_,
                    end2_, keyFn2_);
  }

  Iterator end() const {
    return Iterator(end0_, end0_, keyFn0_, end1_, end1_, keyFn1_, end2_, end2_,
                    keyFn2_);
  }

private:
  Iterator0 begin0_;
  Iterator0 end0_;
  KeyFn0 keyFn0_;

  Iterator1 begin1_;
  Iterator1 end1_;
  KeyFn1 keyFn1_;

  Iterator2 begin2_;
  Iterator2 end2_;
  KeyFn2 keyFn2_;
};

template <typename Sequence0, typename KeyFn0, typename Sequence1,
          typename KeyFn1, typename Sequence2, typename KeyFn2>
GroupBy3<typename Sequence0::const_iterator, KeyFn0,
         typename Sequence1::const_iterator, KeyFn1,
         typename Sequence2::const_iterator, KeyFn2>
groupBy(const Sequence0 &sequence0, KeyFn0 keyFn0, const Sequence1 &sequence1,
        KeyFn1 keyFn1, const Sequence2 &sequence2, KeyFn2 keyFn2) {
  return {sequence0.begin(), sequence0.end(), keyFn0,
          sequence1.begin(), sequence1.end(), keyFn1,
          sequence2.begin(), sequence2.end(), keyFn2};
}


} // end namespace htm

#endif // NTA_GROUPBY_HPP
