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

/** @file
 * Declarations for math functions
 */

#ifndef NTA_MATH_FUNCTIONS_HPP
#define NTA_MATH_FUNCTIONS_HPP

#include <nupic/utils/Log.hpp> // For NTA_ASSERT

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

namespace nupic {

// TODO: replace other functions by boost/math

static const double pi = 3.14159265358979311600e+00;

//--------------------------------------------------------------------------------
template <typename T> inline T lgamma(T x) { return boost::math::lgamma(x); }

//--------------------------------------------------------------------------------
template <typename T> inline T digamma(T x) { return boost::math::digamma(x); }

//--------------------------------------------------------------------------------
template <typename T> inline T beta(T x, T y) {
  return boost::math::beta(x, y);
}

//--------------------------------------------------------------------------------
template <typename T> inline T erf(T x) { return boost::math::erf(x); }

//--------------------------------------------------------------------------------
double fact(unsigned long n) {
  static double a[171];
  static bool init = true;

  if (init) {
    init = false;
    a[0] = 1.0;
    for (size_t i = 1; i != 171; ++i)
      a[i] = i * a[i - 1];
  }

  if (n < 171)
    return a[n];
  else
    return exp(lgamma(n + 1.0));
}

//--------------------------------------------------------------------------------
double lfact(unsigned long n) {
  static double a[2000];
  static bool init = true;

  if (init) {
    for (size_t i = 0; i != 2000; ++i)
      a[i] = lgamma(i + 1.0);
  }

  if (n < 2000)
    return a[n];
  else
    return lgamma(n + 1.0);
}

//--------------------------------------------------------------------------------
double binomial(unsigned long n, unsigned long k) {
  {
    NTA_ASSERT(k <= n) << "binomial: Wrong arguments: n= " << n << " k= " << k;
  }

  if (n < 171)
    return floor(0.5 + fact(n) / (fact(k) * fact(n - k)));
  else
    return floor(0.5 + exp(lfact(n) - lfact(k) - lfact(n - k)));
}

//--------------------------------------------------------------------------------
}; // namespace nupic

#endif // NTA_MATH_FUNCTIONS_HPP
