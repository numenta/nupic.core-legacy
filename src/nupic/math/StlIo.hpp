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

/** @file STL IO
 * This file contains functions to print out and save/load various STL data
 * structures.
 */

#ifndef NTA_STL_IO_HPP
#define NTA_STL_IO_HPP

#include <list>
#include <map>
#include <set>
#include <vector>

#include <iomanip>
#include <iostream>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

//--------------------------------------------------------------------------------
// IO CONTROL AND MANIPULATORS
//--------------------------------------------------------------------------------

struct IOControl { //TODO remove other, all of IOControl

  int abbr;           // shorten long vectors output
  bool output_n_elts; // output vector size at beginning

  bool pair_paren;      // put parens around pairs in vector of pairs
  const char *pair_sep; // put separator between pair.first and pair.second

  inline IOControl(int a = -1, bool s = true, bool pp = false,
                   const char *psep = " ")
      : abbr(a), output_n_elts(s), pair_paren(pp), pair_sep(psep) 
  {}

  inline void reset() {
    abbr = -1;
    output_n_elts = true;
    pair_paren = false;
    pair_sep = " ";
  }
};

extern IOControl io_control;

template <typename CharT, typename Traits, typename T>
inline std::basic_ostream<CharT, Traits> &operator,(
    std::basic_ostream<CharT, Traits> &out_stream, const T &a) {
  return out_stream << ' ' << a;
}

template <typename CharT, typename Traits, typename T>
inline std::basic_istream<CharT, Traits> &operator,(
    std::basic_istream<CharT, Traits> &in_stream, T &a) {
  return in_stream >> a;
}

template <typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits> &operator,(
    std::basic_ostream<CharT, Traits> &out_stream,
    std::basic_ostream<CharT, Traits> &(*pf)(
        std::basic_ostream<CharT, Traits> &)) {
  pf(out_stream);
  return out_stream;
}


struct abbr {
  int n;
  inline abbr(int _n) : n(_n) {}
};

template <typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits> &
operator<<(std::basic_ostream<CharT, Traits> &out_stream, abbr s) {
  io_control.abbr = s.n;
  return out_stream;
}

struct debug {
  int n;
  inline debug(int _n = -1) : n(_n) {}
};

template <typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits> &
operator<<(std::basic_ostream<CharT, Traits> &out_stream, debug d) {
  io_control.abbr = d.n;
  io_control.output_n_elts = false;
  io_control.pair_sep = ",";
  io_control.pair_paren = true;
  return out_stream;
}

//--------------------------------------------------------------------------------
// BINARY PERSISTENCE
//--------------------------------------------------------------------------------
template <typename It>
inline void binary_save(std::ostream &out_stream, It begin, It end) {
  typedef typename std::iterator_traits<It>::value_type value_type;
  size_t size = (size_t)(end - begin);
  if (size > 0) {
    char *ptr = (char *)&*begin;
    out_stream.write(ptr, (std::streamsize)size * sizeof(value_type));
  }
}

//--------------------------------------------------------------------------------
template <typename It>
inline void binary_load(std::istream &in_stream, It begin, It end) {
  typedef typename std::iterator_traits<It>::value_type value_type;
  size_t size = (size_t)(end - begin);
  if (size > 0) {
    char *ptr = (char *)&*begin;
    in_stream.read(ptr, (std::streamsize)size * sizeof(value_type));
  }
}

//--------------------------------------------------------------------------------
template <typename T>
inline void binary_save(std::ostream &out_stream, const std::vector<T> &v) {
  nupic::binary_save(out_stream, v.begin(), v.end());
}

//--------------------------------------------------------------------------------
template <typename T>
inline void binary_load(std::istream &in_stream, std::vector<T> &v) {
  nupic::binary_load(in_stream, v.begin(), v.end());
}

//--------------------------------------------------------------------------------
// STL STREAMING OPERATORS
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// std::pair
//--------------------------------------------------------------------------------
template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &out_stream,
                                const std::pair<T1, T2> &p) {
  if (io_control.pair_paren)
    out_stream << "(";
  out_stream << p.first;
  out_stream << io_control.pair_sep;
  out_stream << p.second;
  if (io_control.pair_paren)
    out_stream << ")";
  return out_stream;
}

//--------------------------------------------------------------------------------
template <typename T1, typename T2>
inline std::istream &operator>>(std::istream &in_stream, std::pair<T1, T2> &p) {
  in_stream >> p.first >> p.second;
  return in_stream;
}

//--------------------------------------------------------------------------------
// std::vector
//--------------------------------------------------------------------------------
template <typename T, bool> struct vector_loader {
  inline void load(size_t, std::istream &, std::vector<T> &);
};

//--------------------------------------------------------------------------------
/**
 * Partial specialization of above functor for primitive types.
 */
template <typename T> struct vector_loader<T, true> {
  inline void load(size_t n, std::istream &in_stream, std::vector<T> &v) {
      for (size_t i = 0; i != n; ++i)
        in_stream >> v[i];
  }
};

// declartion of >> which is used in the following function. Avoid lookup error
template <typename T>
inline std::istream &operator>>(std::istream &in_stream, std::vector<T> &v);
//--------------------------------------------------------------------------------
/**
 * Partial specialization for non-primitive types.
 */
template <typename T> struct vector_loader<T, false> {
  inline void load(size_t n, std::istream &in_stream, std::vector<T> &v) {
    for (size_t i = 0; i != n; ++i)
      in_stream >> v[i];
  }
};

//--------------------------------------------------------------------------------
/**
 * Factory that will instantiate the right functor to call depending on whether
 * T is a primitive type or not.
 */
template <typename T>
inline void vector_load(size_t n, std::istream &in_stream, std::vector<T> &v) {
  vector_loader<T, std::is_fundamental<T>::value > loader;
  loader.load(n, in_stream, v);
}

//--------------------------------------------------------------------------------
template <typename T, bool> struct vector_saver {
  inline void save(size_t n, std::ostream &out_stream, const std::vector<T> &v);
};

//--------------------------------------------------------------------------------
/**
 * Partial specialization for primitive types.
 */
template <typename T> struct vector_saver<T, true> {
  inline void save(size_t n, std::ostream &out_stream,
                   const std::vector<T> &v) {
    if (io_control.output_n_elts) out_stream << n << ' ';

    if (io_control.abbr > 0) {
      n = std::min((size_t)io_control.abbr, n);
    }

    for (size_t i = 0; i != n; ++i) out_stream << v[i] << ' ';

    if (io_control.abbr > 0 && n < v.size()) {
      size_t rest = v.size() - n;
      out_stream << "[+" << rest << "/" << count_non_zeros(v) << "]";
    }
  }
};

// declartion of << which is used in the following function. Avoid lookup error.
template <typename T>
inline std::ostream &operator<<(std::ostream &out_stream,
                                const std::vector<T> &v);
//--------------------------------------------------------------------------------
/**
 * Partial specialization for non-primitive types.
 */
template <typename T> struct vector_saver<T, false> {
  inline void save(size_t n, std::ostream &out_stream,
                   const std::vector<T> &v) {
    if (io_control.output_n_elts)
      out_stream << n << ' ';

    if (io_control.abbr > 0)
      n = std::min((size_t)io_control.abbr, n);

    for (size_t i = 0; i != n; ++i)
      out_stream << v[i] << ' ';

    if (io_control.abbr > 0 && n < v.size()) {
      size_t rest = v.size() - n;
      out_stream << "[+" << rest << "/" << count_non_zeros(v) << "]";
    }
  }
};

//--------------------------------------------------------------------------------
/**
 * Factory that will instantiate the right functor to call depending on whether
 * T is a primitive type or not.
 */
template <typename T>
inline void vector_save(size_t n, std::ostream &out_stream,
                        const std::vector<T> &v) {
  vector_saver<T, std::is_fundamental<T>::value> saver;
  saver.save(n, out_stream, v);
}

//--------------------------------------------------------------------------------
/**
 * Saves the size of the vector.
 */
template <typename T>
inline std::ostream &operator<<(std::ostream &out_stream,
                                const std::vector<T> &v) {
  vector_save(v.size(), out_stream, v);
  return out_stream;
}

//--------------------------------------------------------------------------------
/**
 * Reads in size of the vector, and redimensions it, except if we are reading
 * a sparse binary vector.
 */
template <typename T>
inline std::istream &operator>>(std::istream &in_stream, std::vector<T> &v) {
  size_t n = 0;
  in_stream >> n;
  v.resize(n);
  vector_load(n, in_stream, v);
  return in_stream;
}


//--------------------------------------------------------------------------------
// std::set
//--------------------------------------------------------------------------------
template <typename T1>
inline std::ostream &operator<<(std::ostream &out_stream,
                                const std::set<T1> &m) {
  typename std::set<T1>::const_iterator it = m.begin(), end = m.end();

  while (it != end) {
    out_stream << *it << ' ';
    ++it;
  }

  return out_stream;
}

//--------------------------------------------------------------------------------
// std::map
// Warning: This will not handle elements containing whitespace.
//--------------------------------------------------------------------------------
template <typename T1, typename T2>
  inline std::ostream& operator<<(std::ostream& out_stream, const std::map<T1, T2>& m)
  {
    out_stream << "[ " << m.size() << "\n";

  typename std::map<T1, T2>::const_iterator it = m.begin(), end = m.end();

  while (it != end) {
    out_stream << it->first << ' ' << it->second << ' ';
    ++it;
  }
    out_stream << "]\n";
  return out_stream;
}

  //--------------------------------------------------------------------------------
  template <typename T1, typename T2>
  inline std::istream& operator>>(std::istream& in_stream, std::map<T1, T2>& m)
  {
    std::string tag;
    size_t size = 0;

    in_stream >> tag;
    NTA_CHECK(tag == "[");
    in_stream >> size;

    m.clear();
    for (size_t i = 0; i != size; ++i) {
      T1 k; T2 v;
      in_stream >> k >> v;
      m.insert(std::make_pair(k, v));
    }
    in_stream >> tag;
    NTA_CHECK(tag == "]") << "Expected a closing ']' after map object.";
    in_stream.ignore(1);

    return in_stream;
  }


//--------------------------------------------------------------------------------
} // end namespace nupic
#endif // NTA_STL_IO_HPP
