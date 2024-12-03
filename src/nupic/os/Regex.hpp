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

/** @file */

#ifndef NTA_REGEX_HPP
#define NTA_REGEX_HPP

//----------------------------------------------------------------------

#include <string>

//----------------------------------------------------------------------

namespace nupic {
namespace regex {
bool match(const std::string &re, const std::string &text);
}
} // namespace nupic

#endif // NTA_REGEX_HPP
