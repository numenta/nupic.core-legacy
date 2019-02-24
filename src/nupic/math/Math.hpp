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

namespace nupic {

/**
 *  Epsilon is defined for the whole math and algorithms of the Numenta
 * Platform, independently of the concrete type chosen to handle floating point
 * numbers. numeric_limits<float>::epsilon() == 1.19209e-7
 *   numeric_limits<double>::epsilon() == 2.22045e-16
 */
static const nupic::Real32 Epsilon = nupic::Real(1e-6);
}
#endif
