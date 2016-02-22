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

#ifndef NTA_ENCODERS_SCALAR
#define NTA_ENCODERS_SCALAR

#include <nupic/encoders/Base.hpp>

namespace nupic
{
  class ScalarEncoder : public Encoder
  {
  public:
    ScalarEncoder(int w, double minval, double maxval, int n, double radius,
                  double resolution, bool clipInput);
    ~ScalarEncoder() override;

    virtual void encodeIntoArray(const ArrayBase & input, UInt output[], bool learn) override;
    virtual int getWidth() const override { return n_; }

  private:
    int w_;
    int n_;
    double minval_;
    double maxval_;
    double resolution_;
    bool clipInput_;
  };

  class PeriodicScalarEncoder : public Encoder
  {
  public:
    PeriodicScalarEncoder(int w, double minval, double maxval, int n,
                          double radius, double resolution);
    ~PeriodicScalarEncoder() override;

    virtual void encodeIntoArray(const ArrayBase & input, UInt output[], bool learn) override;
    virtual int getWidth() const override { return n_; }

  private:
    int w_;
    int n_;
    double minval_;
    double maxval_;
    double resolution_;
  };
}

#endif // NTA_ENCODERS_SCALAR
