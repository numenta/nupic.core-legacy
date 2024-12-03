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

#ifndef NTA_FRACTION_HPP
#define NTA_FRACTION_HPP

#include <ostream>

namespace nupic {
class Fraction {
private:
  int numerator_, denominator_;
  // arbitrary cutoff -- need to fix overflow handling. 64-bits everywhere?
  const static int overflowCutoff = 10000000;

public:
  Fraction(int _numerator, int _denominator);
  Fraction(int _numerator);
  Fraction();

  bool isNaturalNumber();

  int getNumerator();
  int getDenominator();

  void setNumerator(int _numerator);
  void setDenominator(int _denominator);
  void setFraction(int _numerator, int _denominator);

  static unsigned int computeGCD(int a, int b);
  static unsigned int computeLCM(int a, int b);

  void reduce();

  Fraction operator*(const Fraction &rhs);
  Fraction operator*(const int rhs);
  friend Fraction operator/(const Fraction &lhs, const Fraction &rhs);
  friend Fraction operator-(const Fraction &lhs, const Fraction &rhs);
  Fraction operator+(const Fraction &rhs);
  Fraction operator%(const Fraction &rhs);
  bool operator<(const Fraction &rhs);
  bool operator>(const Fraction &rhs);
  bool operator<=(const Fraction &rhs);
  bool operator>=(const Fraction &rhs);
  friend bool operator==(Fraction lhs, Fraction rhs);
  friend std::ostream &operator<<(std::ostream &out, Fraction rhs);

  static Fraction fromDouble(double value, unsigned int tolerance = 10000);
  double toDouble();
};
} // namespace nupic

#endif // NTA_FRACTION_HPP
