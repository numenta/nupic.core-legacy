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

#include <cmath>
#include <cstdlib> //abs!
#include <limits>
#include <ostream>
#include <vector>

#include <nupic/types/Exception.hpp>
#include <nupic/types/Fraction.hpp>

namespace nupic {
Fraction::Fraction(int _numerator, int _denominator)
    : numerator_(_numerator), denominator_(_denominator) {
  if (_denominator == 0) {
    throw Exception(__FILE__, __LINE__,
                    "Fraction - attempt to create with invalid zero valued "
                    "denominator");
  }

  // can't use abs() because abs(std::numeric_limits<int>::min()) == -2^31
  if ((numerator_ > overflowCutoff) || (numerator_ < -overflowCutoff) ||
      (denominator_ > overflowCutoff) || (denominator_ < -overflowCutoff)) {
    throw Exception(__FILE__, __LINE__, "Fraction - integer overflow.");
  }
}

Fraction::Fraction(int _numerator) : numerator_(_numerator), denominator_(1) {
  // can't use abs() because abs(std::numeric_limits<int>::min()) == -2^31
  if ((numerator_ > overflowCutoff) || (numerator_ < -overflowCutoff)) {
    throw Exception(__FILE__, __LINE__, "Fraction - integer overflow.");
  }
}

Fraction::Fraction() : numerator_(0), denominator_(1) {}

bool Fraction::isNaturalNumber() {
  return (((numerator_ % denominator_) == 0) &&
          ((*this > 0) || (numerator_ == 0)));
}

int Fraction::getNumerator() { return numerator_; }

int Fraction::getDenominator() { return denominator_; }

void Fraction::setNumerator(int _numerator) { numerator_ = _numerator; }

void Fraction::setDenominator(int _denominator) {
  if (_denominator == 0) {
    throw Exception(__FILE__, __LINE__,
                    "Fraction - attempt to set an invalid zero valued "
                    "denominator");
  }
  denominator_ = _denominator;
}

void Fraction::setFraction(int _numerator, int _denominator) {
  numerator_ = _numerator;
  denominator_ = _denominator;
  if (_denominator == 0) {
    throw Exception(__FILE__, __LINE__,
                    "Fraction - attempt to set an invalid zero valued "
                    "denominator");
  }
}

unsigned int Fraction::computeGCD(int a, int b) {
  unsigned int x, y, r;

  if (a == 0) {
    if (b > 0) {
      return b;
    } else {
      return 1;
    }
  } else if (b == 0) {
    if (a > 0) {
      return a;
    } else {
      return 1;
    }
  }

  // Euclid's algorithm
  a > b ? (x = abs(a), y = abs(b)) : (x = abs(b), y = abs(a));

  r = x % y;

  while (r != 0) {
    x = y;
    y = r;

    r = x % y;
  }

  return y;
}

unsigned int Fraction::computeLCM(int a, int b) {
  int lcm = a * b / ((int)computeGCD(a, b));
  if (lcm < 0) {
    lcm = 0;
  }
  return lcm;
}

void Fraction::reduce() {
  if (numerator_ == 0) {
    denominator_ = 1;
  } else {
    unsigned int m = computeGCD(numerator_, denominator_);

    numerator_ /= (int)m;
    denominator_ /= (int)m;
  }
  if (denominator_ < 0) {
    numerator_ *= -1;
    denominator_ *= -1;
  }
}

Fraction Fraction::operator*(const Fraction &rhs) {
  return Fraction(numerator_ * rhs.numerator_, denominator_ * rhs.denominator_);
}

Fraction Fraction::operator*(const int rhs) {
  return Fraction(numerator_ * rhs, denominator_);
}

Fraction operator/(const Fraction &lhs, const Fraction &rhs) {
  if (rhs.numerator_ == 0) {
    throw Exception(__FILE__, __LINE__, "Fraction - division by zero error");
  }

  return Fraction(lhs.numerator_ * rhs.denominator_,
                  lhs.denominator_ * rhs.numerator_);
}

Fraction operator-(const Fraction &lhs, const Fraction &rhs) {
  int num, lcm;

  lcm = Fraction::computeLCM(lhs.denominator_, rhs.denominator_);
  num = lhs.numerator_ * (lcm / lhs.denominator_) -
        rhs.numerator_ * (lcm / rhs.denominator_);

  return Fraction(num, lcm);
}

Fraction Fraction::operator+(const Fraction &rhs) {
  int num, den;

  den = computeLCM(denominator_, rhs.denominator_);
  num =
      den / denominator_ * numerator_ + den / rhs.denominator_ * rhs.numerator_;

  return Fraction(num, den);
}

Fraction Fraction::operator%(const Fraction &rhs) {
  // a/b % c/d = (ad % bc) / bc. gives output with same sign as a/b
  if (rhs.numerator_ == 0) {
    throw Exception(__FILE__, __LINE__, "Fraction - division by zero error");
    return Fraction(0, 1);
  }

  return Fraction((rhs.denominator_ * numerator_) %
                      (denominator_ * rhs.numerator_),
                  denominator_ * rhs.denominator_);
}

bool Fraction::operator<(const Fraction &rhs) {
  // a/b < c/d if (ad)/(bd) < (bc)/(bd), i.e. if a*d < b*c
  bool negLHS = (denominator_ < 0);
  bool negRHS = (rhs.denominator_ < 0);
  if ((negLHS || negRHS) && !(negLHS && negRHS)) {
    return ((numerator_ * rhs.denominator_) > (denominator_ * rhs.numerator_));
  } else {
    return ((numerator_ * rhs.denominator_) < (denominator_ * rhs.numerator_));
  }
}

bool Fraction::operator>(const Fraction &rhs) {
  // a/b > c/d if (ad)/(bd) > (bc)/(bd), i.e. if a*d > b*c
  bool negLHS = (denominator_ < 0);
  bool negRHS = (rhs.denominator_ < 0);
  if ((negLHS || negRHS) && !(negLHS && negRHS)) {
    return ((numerator_ * rhs.denominator_) < (denominator_ * rhs.numerator_));
  } else {
    return ((numerator_ * rhs.denominator_) > (denominator_ * rhs.numerator_));
  }
}

bool Fraction::operator<=(const Fraction &rhs) {
  return ((Fraction(numerator_, denominator_) < rhs) ||
          (Fraction(numerator_, denominator_) == rhs));
}

bool Fraction::operator>=(const Fraction &rhs) {
  return ((Fraction(numerator_, denominator_) > rhs) ||
          (Fraction(numerator_, denominator_) == rhs));
}

bool operator==(Fraction lhs, Fraction rhs) {
  lhs.reduce();
  rhs.reduce();

  return (lhs.numerator_ == rhs.numerator_ &&
          lhs.denominator_ == rhs.denominator_);
}

std::ostream &operator<<(std::ostream &out, Fraction rhs) {
  rhs.reduce();

  if (rhs.denominator_ == 1) {
    out << rhs.numerator_;
  } else {
    out << rhs.numerator_ << "/" << rhs.denominator_;
  }

  return out;
}

// Recovers a fraction representation of a provided double by building
// a continued fraction and stopping when a continuation component's
// denominator exceeds the provided tolerance.
Fraction Fraction::fromDouble(double value, unsigned int tolerance) {
  std::vector<int> components;
  int component, numerator_, denominator_;
  double continuation;
  bool isNegative;

  if (value < 0) {
    isNegative = true;
    continuation = -value;
  } else {
    isNegative = false;
    continuation = value;
  }

  // use arbitrary cutoff for integer values set in Fraction.hpp
  if (std::abs(value) > overflowCutoff) {
    throw Exception(__FILE__, __LINE__,
                    "Fraction - integer overflow for abritrary cutoff.");
  } else if ((std::fabs(value) < 1.0 / overflowCutoff) &&
             (std::fabs(value) > 0)) {
    throw Exception(__FILE__, __LINE__,
                    "Fraction - integer underflow for arbitrary cutoff.");
  }

  do {
    component = (int)continuation;
    components.push_back(component);
    continuation = 1.0 / (continuation - (double)component);
  } while (continuation < tolerance && components.size() < 100);

  denominator_ = 1;
  numerator_ = components.back();
  components.pop_back();

  while (components.size()) {
    std::swap(numerator_, denominator_);
    numerator_ += denominator_ * components.back();
    components.pop_back();
  }

  if (isNegative) {
    numerator_ *= -1;
  }

  return (Fraction(numerator_, denominator_));
}

double Fraction::toDouble() {
  return ((double)numerator_ / (double)denominator_);
}
} // namespace nupic
