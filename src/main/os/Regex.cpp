/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file 
*/

#include <nta/os/Regex.hpp>
#include <nta/utils/Log.hpp>
#include <regex>

namespace nta
{
  namespace regex
  {
    bool match(const std::string & re, const std::string & text)
    {        
      NTA_CHECK(!re.empty()) << "Empty regular expressions is invalid";
      
      // Make sure the regex will perform an exact match
      std::string exactRegExp;
      if (re[0] != '^')
        exactRegExp += '^';
      exactRegExp += re;
      if (re[re.length()-1] != '$') 
        exactRegExp += '$';
      
	  std::regex r(exactRegExp, std::regex::nosubs);
	  if (std::regex_match(text, r))
		  return true;

	  return false;
    }
  }
}
