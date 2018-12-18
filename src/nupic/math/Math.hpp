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

/** @file
 * Declarations for maths routines
 */

#ifndef NTA_MATH_HPP
#define NTA_MATH_HPP

#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>


#include <nupic/math/Utils.hpp>
#include <nupic/types/Types.hpp>

//--------------------------------------------------------------------------------
/**
 * Macros to make it easier to work with Boost concept checks
 * But we are not going to use boost when using C++17
 */
#if __cplusplus >= 201703L || (defined(_MSC_VER) && _MSC_VER >= 1914)
	// C++17
	#define ASSERT_INPUT_ITERATOR(It)
	#define ASSERT_OUTPUT_ITERATOR(It, T)
	#define ASSERT_UNARY_PREDICATE(UnaryPredicate, Arg1)
	#define ASSERT_UNARY_FUNCTION(UnaryFunction, Ret, Arg1)
	#define ASSERT_BINARY_FUNCTION(BinaryFunction, Ret, Arg1, Arg2)
#else
	#include <boost/concept_check.hpp>

	// Assert that It obeys the STL forward iterator concept
	#define ASSERT_INPUT_ITERATOR(It) \
	  boost::function_requires<boost::InputIteratorConcept<It> >();

	// Assert that It obeys the STL forward iterator concept
	#define ASSERT_OUTPUT_ITERATOR(It, T) \
	  boost::function_requires<boost::OutputIteratorConcept<It, T> >();

	// Assert that UnaryPredicate obeys the STL unary predicate concept
	#define ASSERT_UNARY_PREDICATE(UnaryPredicate, Arg1) \
	  boost::function_requires< \
	      boost::UnaryPredicateConcept<UnaryPredicate, Arg1> >();

	// Assert that UnaryFunction obeys the STL unary function concept
	#define ASSERT_UNARY_FUNCTION(UnaryFunction, Ret, Arg1) \
	  boost::function_requires< \
	      boost::UnaryFunctionConcept<UnaryFunction, Ret, Arg1> >();

	// Assert that BinaryFunction obeys the STL binary function concept
	#define ASSERT_BINARY_FUNCTION(BinaryFunction, Ret, Arg1, Arg2) \
	  boost::function_requires< \
	      boost::BinaryFunctionConcept<BinaryFunction, Ret, Arg1, Arg2> >();
#endif

//--------------------------------------------------------------------------------
namespace nupic {

//--------------------------------------------------------------------------------
// ASSERTIONS
//--------------------------------------------------------------------------------
/**
 * This is used to check that a boolean condition holds, and to send a message
 * on NTA_INFO if it doesn't. It's compiled in only when NTA_ASSERTIONS_ON is
 * true.
 */
inline bool INVARIANT(bool cond, const char *msg) {
#ifdef NTA_ASSERTIONS_ON
  if (!(cond)) {
    NTA_INFO << msg;
    return false;
  }
#endif
  return true;
}

//--------------------------------------------------------------------------------
/**
 * An assert used to validate that a range defined by two iterators is valid,
 * that is: begin <= end. We could say that end < begin means the range is
 * empty, but for debugging purposes, we want to know when the iterators
 * are crossed.
 */
template <typename It>
void ASSERT_VALID_RANGE(It begin, It end, const char *message) {
  NTA_ASSERT(begin <= end) << "Invalid iterators: " << message;
}

//--------------------------------------------------------------------------------
/**
 *  Epsilon is defined for the whole math and algorithms of the Numenta
 * Platform, independently of the concrete type chosen to handle floating point
 * numbers. numeric_limits<float>::epsilon() == 1.19209e-7
 *   numeric_limits<double>::epsilon() == 2.22045e-16
 */
static const nupic::Real Epsilon = nupic::Real(1e-6);

//--------------------------------------------------------------------------------
/**
 * Functions that test for positivity or negativity based on nupic::Epsilon.
 */
template <typename T> inline bool strictlyNegative(const T &a) {
  return a < -nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool strictlyPositive(const T &a) {
  return a > nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool negative(const T &a) {
  return a <= nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool positive(const T &a) {
  return a >= -nupic::Epsilon;
}

//--------------------------------------------------------------------------------
/**
 * A functions that implements the distance to zero function as a functor.
 * Defining argument_type and result_type directly here instead of inheriting
 * from std::unary_function so that we have an easier time in SWIG Python
 * wrapping.
 */
template <typename T> struct DistanceToZero {
  typedef T argument_type;
  typedef T result_type;

  inline T operator()(const T &x) const { return x >= 0 ? x : -x; }
};

//--------------------------------------------------------------------------------
/**
 * A specialization for UInts, where we only need one test (more efficient).
 */
template <> inline UInt DistanceToZero<UInt>::operator()(const UInt &x) const {
  return x;
}

//--------------------------------------------------------------------------------
/**
 * Use this functor if T is guaranteed to be positive only.
 */
template <typename T>
struct DistanceToZeroPositive : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return x; }
};

//--------------------------------------------------------------------------------
/**
 * This computes the distance to 1 rather than to 0.
 */
template <typename T> struct DistanceToOne {
  typedef T argument_type;
  typedef T result_type;

  inline T operator()(const T &x) const {
    return x > (T)1 ? x - (T)1 : (T)1 - x;
  }
};

//--------------------------------------------------------------------------------
/**
 * This functor decides whether a number is almost zero or not, using the
 * platform-wide nupic::Epsilon.
 */
template <typename D> struct IsNearlyZero {
  typedef typename D::result_type value_type;

  D dist_;

  // In the case where D::result_type is integral
  // we convert nupic::Epsilon to zero!
  inline IsNearlyZero() : dist_() {}

  inline IsNearlyZero(const IsNearlyZero &other) : dist_(other.dist_) {}

  inline IsNearlyZero &operator=(const IsNearlyZero &other) {
    if (this != &other)
      dist_ = other.dist_;

    return *this;
  }

  inline bool operator()(const typename D::argument_type &x) const {
    return dist_(x) <= nupic::Epsilon;
  }
};

//--------------------------------------------------------------------------------
/**
 * @b Responsibility:
 *  Tell whether an arithmetic value is zero or not, within some precision,
 *  or whether two values are equal or not, within some precision.
 *
 * @b Parameters:
 *  epsilon: accuracy of the comparison
 *
 * @b Returns:
 *  true if |a| <= epsilon
 *  false otherwise
 *
 * @b Requirements:
 *  T arithmetic
 *  T comparable with operator<= AND operator >=
 *
 * @b Restrictions:
 *  Doesn't compile if T is not arithmetic.
 *  In debug mode, NTA_ASSERTs if |a| > 10
 *  In debug mode, NTA_ASSERTs if a == infinity, quiet_NaN or signaling_NaN
 *
 * @b Notes:
 *  Comparing floating point numbers is a pretty tricky business. Knuth's got
 *  many pages devoted to it in Vol II. Boost::test has a special function to
 *  handle that. One of the problems is that when more bits are allocated to
 *  the integer part as the number gets bigger, there is inherently less
 *  precision in the decimals. But, for comparisons to zero, we can continue
 *  using an absolute epsilon (instead of multiplying epsilon by the number).
 *  In our application, we are anticipating numbers mostly between 0 and 1,
 *  because they represent probabilities.
 *
 *  Not clear why this is namespace std instead of nupic , but FA says there was
 *  a "good, ugly" reason to do it this way that he can't remember. - WCS 0206
 */
template <typename T>
inline bool nearlyZero(const T &a, const T &epsilon = T(nupic::Epsilon)) {
  return a >= -epsilon && a <= epsilon;
}

//--------------------------------------------------------------------------------
template <typename T>
inline bool nearlyEqual(const T &a, const T &b,
                        const T &epsilon = nupic::Epsilon) {
  return nearlyZero((b - a), epsilon);
}

//--------------------------------------------------------------------------------
/**
 * Euclidean modulo function.
 *
 * Returns x % m, but keeps the value positive
 * (similar to Python's modulo function).
 */
inline int emod(int x, int m) {
  int r = x % m;
  if (r < 0)
    return r + m;
  else
    return r;
}

//--------------------------------------------------------------------------------
/**
 * A boolean functor that returns true if the element's selected value is found
 * in the (associative) container (needs to support find).
 *
 * Example:
 * =======
 * typedef std::pair<UInt, UInt> IdxVal;
 * std::list<IdxVal> row;
 * std::set<UInt> alreadyGrouped;
 * row.remove_if(SetIncludes<std::set<UInt>, select1st<IdxVal>
 * >(alreadyGrouped));
 *
 * Will remove from row (a list of pairs) the pairs whose first element is
 * already contained in alreadyGrouped.
 */
template <typename C1, typename Selector, bool f = false> struct IsIncluded {
  IsIncluded(const C1 &container) : sel_(), container_(container) {}

  template <typename T> inline bool operator()(const T &p) const {
    if (f)
      return container_.find(sel_(p)) == container_.end();
    else
      return container_.find(sel_(p)) != container_.end();
  }

  Selector sel_;
  const C1 &container_;
};

//--------------------------------------------------------------------------------
// PAIRS AND TRIPLETS
//--------------------------------------------------------------------------------

/**
 * Lexicographic order:
 * (1,1) < (1,2) < (1,10) < (2,5) < (3,6) < (3,7) ...
 */
template <typename T1, typename T2>
struct lexicographic_2
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    if (a.first < b.first)
      return true;
    else if (a.first == b.first)
      if (a.second < b.second)
        return true;
    return false;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the first member of a pair only:
 * (1, 3.5) < (2, 5.6) < (10, 7.1) < (11, 8.5)
 */
template <typename T1, typename T2>
struct less_1st
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.first < b.first;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the second member of a pair only:
 * (10, 3.5) < (1, 5.6) < (2, 7.1) < (11, 8.5)
 */
template <typename T1, typename T2>
struct less_2nd
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.second < b.second;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the first member of a pair only:
 * (10, 3.5) > (8, 5.6) > (2, 7.1) > (1, 8.5)
 */
template <typename T1, typename T2>
struct greater_1st
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.first > b.first;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the second member of a pair only:
 * (10, 3.5) > (1, 5.6) > (2, 7.1) > (11, 8.5)
 */
template <typename T1, typename T2>
struct greater_2nd
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.second > b.second;
  }
};

//--------------------------------------------------------------------------------
template <typename T1, typename T2>
struct greater_2nd_p : public std::binary_function<bool, std::pair<T1, T2 *>,
                                                   std::pair<T1, T2 *> > {
  inline bool operator()(const std::pair<T1, T2 *> &a,
                         const std::pair<T1, T2 *> &b) const {
    return *(a.second) > *(b.second);
  }
};

//--------------------------------------------------------------------------------
/**
 * A greater 2nd order that breaks ties, useful for debugging.
 */
template <typename T1, typename T2>
struct greater_2nd_no_ties
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    if (a.second > b.second)
      return true;
    else if (a.second == b.second)
      if (a.first < b.first)
        return true;
    return false;
  }
};

//--------------------------------------------------------------------------------
template <typename T1, typename T2, typename RND>
struct greater_2nd_rnd_ties
    : public std::binary_function<bool, std::pair<T1, T2>, std::pair<T1, T2> > {
  RND &rng;

  inline greater_2nd_rnd_ties(RND &_rng) : rng(_rng) {}

  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    T2 val_a = a.second, val_b = b.second;
    if (val_a > val_b)
      return true;
    else if (val_a == val_b)
      return rng.getReal64() >= .5;
    return false;
  }
};

//--------------------------------------------------------------------------------
// A class used to work with lists of non-zeros represented in i,j,v format
//--------------------------------------------------------------------------------
/**
 * This class doesn't implement any algorithm, it just stores i,j and v.
 */
template <typename T1, typename T2> class ijv {
  typedef T1 size_type;
  typedef T2 value_type;

private:
  size_type i_, j_;
  value_type v_;

public:
  inline ijv() : i_(0), j_(0), v_(0) {}

  inline ijv(size_type i, size_type j, value_type v) : i_(i), j_(j), v_(v) {}

  inline ijv(const ijv &o) : i_(o.i_), j_(o.j_), v_(o.v_) {}

  inline ijv &operator=(const ijv &o) {
    i_ = o.i_;
    j_ = o.j_;
    v_ = o.v_;
    return *this;
  }

  inline size_type i() const { return i_; }
  inline size_type j() const { return j_; }
  inline value_type v() const { return v_; }
  inline void i(size_type ii) { i_ = ii; }
  inline void j(size_type jj) { j_ = jj; }
  inline void v(value_type vv) { v_ = vv; }

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct lexicographic : public std::binary_function<bool, ijv, ijv> {
    inline bool operator()(const ijv &a, const ijv &b) const {
      if (a.i() < b.i())
        return true;
      else if (a.i() == b.i())
        if (a.j() < b.j())
          return true;
      return false;
    }
  };

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct less_value : public std::binary_function<bool, ijv, ijv> {
    inline bool operator()(const ijv &a, const ijv &b) const {
      return a.v() < b.v();
    }
  };

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct greater_value : public std::binary_function<bool, ijv, ijv> {
    inline bool operator()(const ijv &a, const ijv &b) const {
      return a.v() > b.v();
    }
  };
};

//--------------------------------------------------------------------------------
/**
 * These templates allow the implementation to use the right function depending
 * on the data type, which yield significant speed-ups when working with floats.
 * They also extend STL's arithmetic operations.
 */
//--------------------------------------------------------------------------------
// Unary functions
//--------------------------------------------------------------------------------

template <typename T> struct Identity : public std::unary_function<T, T> {
  inline T &operator()(T &x) const { return x; }
  inline const T &operator()(const T &x) const { return x; }
};

template <typename T> struct Negate : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return -x; }
};

template <typename T> struct Abs : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return x > 0.0 ? x : -x; }
};

template <typename T> struct Square : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return x * x; }
};

template <typename T> struct Cube : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return x * x * x; }
};

template <typename T> struct Inverse : public std::unary_function<T, T> {
  inline T operator()(const T &x) const { return 1.0 / x; }
};

template <typename T> struct Sqrt : public std::unary_function<T, T> {};

template <> struct Sqrt<float> : public std::unary_function<float, float> {
  inline float operator()(const float &x) const { return sqrtf(x); }
};

template <> struct Sqrt<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const { return sqrt(x); }
};

template <typename T> struct Exp : public std::unary_function<T, T> {};

template <> struct Exp<float> : public std::unary_function<float, float> {
  // On x86_64, there is a bug in glibc that makes expf very slow
  // (more than it should be), so we continue using exp on that
  // platform as a workaround.
  // https://bugzilla.redhat.com/show_bug.cgi?id=521190
  // To force the compiler to use exp instead of expf, the return
  // type (and not the argument type!) needs to be double.
  inline float operator()(const float &x) const { return expf(x); }
};

template <> struct Exp<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const { return exp(x); }
};

template <typename T> struct Log : public std::unary_function<T, T> {};

template <> struct Log<float> : public std::unary_function<float, float> {
  inline float operator()(const float &x) const { return logf(x); }
};

template <> struct Log<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const { return log(x); }
};

template <typename T> struct Log2 : public std::unary_function<T, T> {};

template <> struct Log2<float> : public std::unary_function<float, float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)(log(x) / log(2.0));
#else
    return log2f(x);
#endif
  }
};

template <> struct Log2<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(x) / log(2.0);
#else
    return log2(x);
#endif
  }
};

template <typename T> struct Log10 : public std::unary_function<T, T> {};

template <> struct Log10<float> : public std::unary_function<float, float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)(log(x) / log(10.0));
#else
    return log10f(x);
#endif
  }
};

template <> struct Log10<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(x) / log(10.0);
#else
    return log10(x);
#endif
  }
};

template <typename T> struct Log1p : public std::unary_function<T, T> {};

template <> struct Log1p<float> : public std::unary_function<float, float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)log(1.0 + x);
#else
    return log1pf(x);
#endif
  }
};

template <> struct Log1p<double> : public std::unary_function<double, double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(1.0 + x);
#else
    return log1p(x);
#endif
  }
};

/**
 * Numerical approximation of derivative.
 * Error is h^4 y^5/30.
 */
template <typename Float, typename F>
struct Derivative : public std::unary_function<Float, Float> {
  Derivative(const F &f) : f_(f) {}

  F f_;

  /**
   * Approximates the derivative of F at x.
   */
  inline const Float operator()(const Float &x) const {
    const Float h = nupic::Epsilon;
    return (-f_(x + 2 * h) + 8 * f_(x + h) - 8 * f_(x - h) + f_(x - 2 * h)) /
           (12 * h);
  }
};

//--------------------------------------------------------------------------------
// Binary functions
//--------------------------------------------------------------------------------
template <typename T> struct Assign : public std::binary_function<T, T, T> {
  inline T operator()(T &x, const T &y) const {
    x = y;
    return x;
  }
};

template <typename T> struct Plus : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T> struct Minus : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x - y; }
};

template <typename T> struct Multiplies : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T> struct Divides : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x / y; }
};

template <typename T> struct Pow : public std::binary_function<T, T, T> {};

template <>
struct Pow<float> : public std::binary_function<float, float, float> {
  inline float operator()(const float &x, const float &y) const {
    return powf(x, y);
  }
};

template <>
struct Pow<double> : public std::binary_function<double, double, double> {
  inline double operator()(const double &x, const double &y) const {
    return pow(x, y);
  }
};

template <typename T> struct Logk : public std::binary_function<T, T, T> {};

template <>
struct Logk<float> : public std::binary_function<float, float, float> {
  inline float operator()(const float &x, const float &y) const {
    return logf(x) / logf(y);
  }
};

template <>
struct Logk<double> : public std::binary_function<double, double, double> {
  inline double operator()(const double &x, const double &y) const {
    return log(x) / log(y);
  }
};

template <typename T> struct Max : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x > y ? x : y; }
};

template <typename T> struct Min : public std::binary_function<T, T, T> {
  inline T operator()(const T &x, const T &y) const { return x < y ? x : y; }
};

//--------------------------------------------------------------------------------
/**
 * This is a 'C++ function object' or Functor.  An object that can be passed
 * as if it were a C function. It is created by having a class containing an
 * overload of the ( ) operator.
 *
 * The std::unary_function is deprecated in C++11 and removed in C++17.
 *
 * Compose two unary functions.
 */
//template <typename F1, typename F2>
//struct unary_compose : public std::unary_function<typename F1::argument_type,
//                                                  typename F2::result_type> {
//  typedef typename F1::argument_type argument_type;
//  typedef typename F2::result_type result_type;
template <typename F1, typename F2> struct unary_compose {
  typedef typename F1::argument_type argument_type;
  typedef typename F2::result_type result_type;

  F1 f1;
  F2 f2;

  inline result_type operator()(const argument_type &x) const {
    return f2(f1(x));
  }
};

//--------------------------------------------------------------------------------
/**
 * This is a 'C++ function object' or Functor.  An object that can be passed
 * as if it were a C function. It is created by having a class containing an
 * overload of the ( ) operator.
 *
 * TODO: The std::binary_function is deprecated in C++11 and removed in C++17.
 *
 * Compose an order predicate and a binary selector, so that we can write:
 * sort(x.begin(), x.end(), compose<less<float>, select2nd<pair<int, float> >
 * >()); to sort pairs in increasing order of their second element.
 */
//template <typename O, typename S>
//struct predicate_compose
//    : public std::binary_function<typename S::argument_type,
//                                  typename S::argument_type, bool> {
//  typedef bool result_type;
//  typedef typename S::argument_type argument_type;
template <typename O, typename S> struct predicate_compose {
  typedef typename S::argument_type argument_type;

  O o;  // operation function object
  S s;  // selection function object

  inline bool operator()(const argument_type &x,  const argument_type &y) const {
    return o(s(x), s(y));
  }
};

//--------------------------------------------------------------------------------
/**
 *  When dividing by a value less than min_exponent10, inf will be generated.
 *   numeric_limits<float>::min_exponent10 = -37
 *   numeric_limits<double>::min_exponent10 = -307
 */
template <typename T> inline bool isSafeForDivision(const T &x) {
  Log<T> log_f;
  return log_f(x) >= std::numeric_limits<T>::min_exponent10;
}

//--------------------------------------------------------------------------------
/**
 * Returns the value passed in or a threshold if the value is >= threshold.
 */
template <typename T> struct ClipAbove : public std::unary_function<T, T> {
  inline ClipAbove(const T &val) : val_(val) {}

  inline ClipAbove(const ClipAbove &c) : val_(c.val_) {}

  inline ClipAbove &operator=(const ClipAbove &c) {
    if (this != &c)
      val_ = c.val_;

    return *this;
  }

  inline T operator()(const T &x) const { return x >= val_ ? val_ : x; }

  T val_;
};

//--------------------------------------------------------------------------------
/**
 * Returns the value passed in or a threshold if the value is < threshold.
 */
template <typename T> struct ClipBelow : public std::unary_function<T, T> {
  inline ClipBelow(const T &val) : val_(val) {}

  inline ClipBelow(const ClipBelow &c) : val_(c.val_) {}

  inline ClipBelow &operator=(const ClipBelow &c) {
    if (this != &c)
      val_ = c.val_;

    return *this;
  }

  inline T operator()(const T &x) const { return x < val_ ? val_ : x; }

  T val_;
};

//--------------------------------------------------------------------------------

}; // namespace nupic

#endif // NTA_MATH_HPP
