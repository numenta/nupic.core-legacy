
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
 * Interface for the Dimensions class
 */

#ifndef NTA_DIMENSIONS_HPP
#define NTA_DIMENSIONS_HPP

#include <iostream>
#include <sstream>
#include <cctype>
#include <vector>
#include <iterator>   // for stream iterator
#include <numeric>    // for std::accumulate
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/types/Serializable.hpp>


namespace nupic {

class Dimensions : public std::vector<UInt>, public Serializable {
public:
  /**
   * Create a new Dimensions object.
   * The dimension in index 0 is the one that moves fastest while iterating.
   * in 2D coordinates, x,y; the x is dimension[0], y is dimension[1].
   * @note Default dimensions are unspecified, see isUnspecified()
   *       Dimensions of size=1 and value [0] = 0 means "not known yet", see isDontCare()
   */
  Dimensions() {};
  Dimensions(UInt x) { push_back(x); }
  Dimensions(UInt x, UInt y) {  push_back(x); push_back(y); }
  Dimensions(UInt x, UInt y, UInt z) { push_back(x); push_back(y); push_back(z); }
  Dimensions(const std::vector<UInt>& v) : std::vector<UInt>(v){};
  Dimensions(const Dimensions& d)  : std::vector<UInt>(d){};

  /**
   * @returns  The count of cells in the grid which is the product of the sizes of
   * the dimensions.
   */
  size_t getCount() const { return((size() > 0) ? std::accumulate(begin(), end(), 1, std::multiplies<UInt>()) : 0);}


  /**
   *
   * There are two "special" values for dimensions:
   *
   * * Dimensions of `[]` (`dims.size()==0`) means "empty dimensions" aka
   *         "unspecified", see isUnspecified() 
   * * Dimensions of `[0]`  (`dims.size()==1 && dims[0] == 0`) means
   *         "in process of being specified but not yet resolved.", see isDontcare()
   *
   * The states that a Dimensions object can have are:
   *
   * * Unspecified - empty; Everything starts out as unspecified.
   *
   * * Dont Care   - We are in the process of setting all dimensions. 
   *                 We have checked direct explicit configuration,
   *                 and trying implied configuration. Not yet resolved.
   *
   *                 For example, if we looked at an input and checked that 
	 *                 it was not configured with a dimension we can mark it
   *                 as isDontCare so that later when we determine the 
   *                 dimensions of the connected output we know that it 
   *                 can also assign it to the input.
   *
   * * Specified   - We have a good dimension. We have at least one dimension.
   *                 It's not the opposite of isUnspecified()!
   *
   * * Invalid     - Some dimension is 0 although Dontcare state is valid.
   *
   * There is a function to check for each of these states.
   */
  static const int DONTCARE = 0;
  bool isUnspecified() const { return(size() == 0); }
  bool isDontcare()    const { return(size() == 1 && at(0) == DONTCARE); }
  bool isInvalid()     const { return(!isDontcare() && getCount() == 0); }
  bool isSpecified()   const { return(getCount() != 0); }

  // TODO:Cereal- when Cereal is fully implmented humanReadable arg can be removed here and in DimensionsTest.
  std::string toString(bool humanReadable = true) const {
    if (isUnspecified()) return "[unspecified]";
    if (isDontcare())    return "[dontcare]";
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size(); i++) {
      if (i)  ss << "," <<at(i);
      else   ss << at(i);
    }
    ss << "] ";
		if (humanReadable && isInvalid()) ss << "(Invalid) ";
    return ss.str();
  }
  /****
  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    ar((std::vector<UInt>&) *this);
  }
  template<class Archive>
  void load_ar(Archive & ar) {
    ar((std::vector<UInt>&) *this);
  }
  ****/
  // TODO:Cereal- remove these two methods when Cereal is fully implmented.
  void save(std::ostream &f) const override {
    size_t n = size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    if (n > 0)
      f.write(reinterpret_cast<const char*>(&at(0)), n * sizeof(at(0)));
  }
  void load(std::istream &f) override {
    size_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    clear();
    if (n > 0) {
      resize(n);
      f.read(reinterpret_cast<char*>(&at(0)), n * sizeof(at(0)));
    }
  }

};
  

  inline std::ostream &operator<<(std::ostream &f, const Dimensions& d) {
    f << d.toString(false) << " ";
    return f;
  }
  inline std::istream &operator>>(std::istream &f, Dimensions& d) { 
    // expected format:    [val, val, val]
    f >> std::ws;  // ignore leading whitespace
    d.clear();
    int c = f.get();
    NTA_CHECK(c == '[') << "Expecting beginning of Dimensions.";
    if (!isdigit(f.peek())) {
      std::string tag;
      f >> tag;
      if (tag == "unspecified]") {
        // leave d empty.
      }
      else if (tag == "dontcare]") {
        d.push_back(0);
      }
    }
    else {
      UInt32 i;
      char buf[50];
      while(isdigit(f.peek())) {
        int j = 0;
        while(isdigit(c = f.get())) {
          buf[j++] = c;
        }
        buf[j] = '\0';
        i = strtoul(buf, nullptr, 0);
        d.push_back(i);
        if (c == ']') break;
        f >> std::ws;  // ignore whitespace
        NTA_CHECK(c == ',') << "Invalid format for Dimensions";
      }
    }
    return f;
  }
  

} // namespace nupic

#endif // NTA_DIMENSIONS_HPP
