#ifndef NTA_PYBIND_SPARSE_TENSOR_HPP
#define NTA_PYBIND_SPARSE_TENSOR_HPP

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
 */


#include <nupic/math/Index.hpp>
#include <nupic/math/Domain.hpp>
#include <nupic/math/SparseTensor.hpp>

#include <vector>
#include <sstream>

//--------------------------------------------------------------------------------
typedef std::vector<nupic::UInt32> TIV;

#define PYSPARSETENSOR_MAX_RANK 20

class PyBindTensorIndex;

inline std::ostream &operator<<(std::ostream &o, const PyBindTensorIndex &j);

//--------------------------------------------------------------------------------
class PyBindTensorIndex
{
  enum { maxRank = PYSPARSETENSOR_MAX_RANK };
  nupic::UInt32 index_[maxRank];
  nupic::UInt32 rank_;

public:
  typedef nupic::UInt32 value_type;
  typedef const nupic::UInt32 *const_iterator;
  typedef nupic::UInt32 *iterator;

  PyBindTensorIndex() : rank_(0) {}

  PyBindTensorIndex(const PyBindTensorIndex &x) : rank_(x.rank_)
  {
    ::memcpy(index_, x.index_, rank_*sizeof(nupic::UInt32));
  }

  PyBindTensorIndex(nupic::UInt32 i) : rank_(1)
  {
    index_[0] = i;
  }

  PyBindTensorIndex(nupic::UInt32 i, nupic::UInt32 j) : rank_(2)
  {
    index_[0] = i;
    index_[1] = j;
  }

  PyBindTensorIndex(nupic::UInt32 i, nupic::UInt32 j, nupic::UInt32 k) : rank_(3)
  {
    index_[0] = i;
    index_[1] = j;
    index_[2] = k;
  }

  PyBindTensorIndex(nupic::UInt32 i, nupic::UInt32 j, nupic::UInt32 k, nupic::UInt32 l) : rank_(4)
  {
    index_[0] = i;
    index_[1] = j;
    index_[2] = k;
    index_[3] = l;
  }

  PyBindTensorIndex(nupic::UInt32 i, nupic::UInt32 j, nupic::UInt32 k, nupic::UInt32 l, nupic::UInt32 m) : rank_(5)
  {
    index_[0] = i;
    index_[1] = j;
    index_[2] = k;
    index_[3] = l;
    index_[4] = m;
  }

  PyBindTensorIndex(nupic::UInt32 i, nupic::UInt32 j, nupic::UInt32 k, nupic::UInt32 l, nupic::UInt32 m, nupic::UInt32 n) : rank_(6)
  {
    index_[0] = i;
    index_[1] = j;
    index_[2] = k;
    index_[3] = l;
    index_[4] = m;
    index_[5] = n;
  }

  PyBindTensorIndex(const TIV &i) : rank_(i.size())
  {
    if (rank_ > maxRank) {
      char errBuf[512];
      snprintf(errBuf, 512, 
               "Tensors may not be constructed of rank greater than %d.", maxRank);
      rank_ = 0;
      throw std::runtime_error(errBuf);
    }
    std::copy(i.begin(), i.end(), index_);
  }

  template<typename T>
  PyBindTensorIndex(int nd, const T *d) : rank_(nd)
  {
    if (nd > maxRank) {
      char errBuf[512];
      snprintf(errBuf, 512, 
               "Tensors may not be constructed of rank greater than %d.", maxRank);
      rank_ = 0;
      throw std::runtime_error(errBuf);
    }
    if(d) std::copy(d, d+nd, index_);
    else std::fill(index_, index_+nd, 0);
  }
  
  PyBindTensorIndex(const PyBindTensorIndex& i1, const PyBindTensorIndex& i2)
    : rank_(i1.rank_ + i2.rank_)
  {
    if (rank_ > maxRank) {
      char errBuf[512];
      snprintf(errBuf, 512, 
               "Tensors may not be constructed of rank greater than %d.", maxRank);
      rank_ = 0;
      throw std::runtime_error(errBuf);
    }

    ::memcpy(index_, i1.index_, i1.rank_*sizeof(nupic::UInt32));
    ::memcpy(index_ + i1.rank_, i2.index_, i2.rank_*sizeof(nupic::UInt32));
  }
  
  PyBindTensorIndex &operator=(const PyBindTensorIndex &x)
  {
    rank_ = x.rank_;
    ::memcpy(index_, x.index_, rank_*sizeof(nupic::UInt32));
    return *this;
  }

  PyBindTensorIndex &operator=(const TIV &i)
  {
    if(i.size() > maxRank) {
      char errBuf[512];
      snprintf(errBuf, 512, 
               "Tensors may not be constructed of rank greater than %d.", maxRank);
      rank_ = 0;
      throw std::runtime_error(errBuf);
    }
    rank_ = i.size();
    std::copy(i.begin(), i.end(), index_);
    return *this;
  }

  nupic::UInt32 size() const { return rank_; }

  nupic::UInt32 operator[](nupic::UInt32 i) const
  {
    if(!(i < rank_)) throw std::invalid_argument("Index out of bounds.");
    return index_[i];
  }
  nupic::UInt32 &operator[](nupic::UInt32 i)
  {
    if(!(i < rank_)) throw std::invalid_argument("Index out of bounds.");
    return index_[i];
  }

  nupic::UInt32 __getitem__(int i) const { if(i < 0) i += rank_; return index_[i]; }
  void __setitem__(int i, nupic::UInt32 d) { if(i < 0) i += rank_; index_[i] = d; }
  nupic::UInt32 __len__() const { return rank_; }

  const nupic::UInt32 *begin() const { return index_; }
  nupic::UInt32 *begin() { return index_; }
  const nupic::UInt32 *end() const { return index_ + rank_; }
  nupic::UInt32 *end() { return index_ + rank_; }

  bool operator==(const PyBindTensorIndex &j) const
  {
    if(rank_ != j.rank_) return false;
    for(nupic::UInt32 i=0; i<rank_; ++i)
      { if(index_[i] != j.index_[i]) return false; }
    return true;
  }
  bool operator!=(const PyBindTensorIndex &j) const { return !((*this) == j); }
  bool operator<(const PyBindTensorIndex &j) const
  {
    const nupic::UInt32 n = rank_ <= j.rank_ ? rank_ : j.rank_;

    for (nupic::UInt32 k = 0; k < n; ++k)
      if (index_[k] < j.index_[k])
        return true;
      else if (index_[k] > j.index_[k])
        return false;
    if(n < j.rank_) return true;
    else return false;
    return false;
  }
  bool __eq__(const PyBindTensorIndex &j) const { return (*this) == j; }
  bool __ne__(const PyBindTensorIndex &j) const { return (*this) != j; }
  //  bool __lt__(const PyBindTensorIndex &j) const { return (*this) < j; }
  bool __gt__(const PyBindTensorIndex &j) const { return j < (*this); }

  bool operator==(const TIV &j) const
  {
    if(size() != j.size()) return false;
    for(nupic::UInt32 i=0; i<rank_; ++i)
      { if(index_[i] != j[i]) return false; }
    return true;
  }
  bool operator!=(const TIV &j) const { return !((*this) == j); }
  bool __eq__(const TIV &j) const { return (*this) == j; }
  bool __ne__(const TIV &j) const { return (*this) != j; }

  std::string __str__() const
  {
    std::stringstream s;
    s << "(";
    nupic::UInt32 n = rank_;
    if(n) {
      s << index_[0];
      for(nupic::UInt32 i=1; i<n; ++i) s << ", " << index_[i];
    }
    s << ")";
    return s.str();
  }

  TIV __getslice__(int i, int j) const
  {
    if(i < 0) i += rank_;
    if(j < 0) j += rank_;
    if(j == 2147483647) j = rank_;
    return TIV(index_ + i, index_ + j);
  }
  
  void __setslice__(int i, int j, const TIV &x)
  {
    if(i < 0) i += rank_;
    if(j < 0) j += rank_;
    if(j == 2147483647) j = rank_;
    std::copy(x.begin(), x.end(), index_ + i);
  }

  TIV asTuple() const
  { return TIV(index_, index_ + rank_); }

  TIV __getstate__() const { return asTuple(); }
};

//--------------------------------------------------------------------------------
inline PyBindTensorIndex concatenate(const PyBindTensorIndex& i1, const PyBindTensorIndex& i2)
{
  return PyBindTensorIndex(i1, i2);
}

//--------------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &o, const PyBindTensorIndex &j) {
  o << "(";
  nupic::UInt32 n = j.size();
  if(n) {
    o << j[0];
    for(nupic::UInt32 i=1; i<n; ++i) o << "," << j[i];
  }
  o << ")";
  return o;
}


//--------------------------------------------------------------------------------

#endif // NTA_PYBIND_SPARSE_TENSOR_HPP
