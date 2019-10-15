/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * David Keeney dkeeney@gmail.com
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
 * --------------------------------------------------------------------- */
#ifndef NTA_VALUE_HPP
#define NTA_VALUE_HPP

#include <htm/utils/Log.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>

namespace htm {



class Value {
public:
  Value();

  // Parse a Yaml or JSON string an assign it to this node in the tree.
  Value &parse(const std::string &yaml_string);

  bool contains(const std::string &key) const;
  size_t size() const;

  // type of value in the Value node
  enum Category { Empty = 0, Scalar, Sequence, Map };
  Category getCategory() const;
  bool isScalar() const;
  bool isSequence() const;
  bool isMap() const;
  bool isEmpty() const; // true if operator[] did not find a value or was not yet assigned to.
  bool isRoot() const;

  // Access
  std::string str() const;   // return copy of raw value as a string
  std::string key() const;   // return copy of raw key as a string
  const char *c_str() const; // return a const pointer to raw value
  std::vector<std::string> getKeys() const;

  Value &operator[](const std::string &key);
  Value &operator[](size_t index);
  const Value &operator[](const std::string &key) const;
  const Value &operator[](size_t index) const;

  template <typename T> inline T as() const {
    if (type_ != Category::Scalar)
      NTA_THROW << "not found. '" << scalar_ << "'";
    T result;
    std::stringstream ss(scalar_);
    ss >> result;
    return result;
  }
  /*
  // A work-around for not allowing Explicit template specializations in the class scope.
  // This is a bug in the GCC compiler before C++17, clang before C++14.  MSVC is ok.
  inline int8_t as<int8_t>() const     { return asInt8();   }
  inline int16_t as<int16_t>() const   { return asInt16();  }
  inline uint16_t as<uint16_t>() const { return asUInt16(); }
  inline int32_t as<int32_t>() const   { return asInt32();  }
  inline uint32_t as<uint32_t>() const { return asUInt32(); }
  inline int64_t as<int64_t>() const   { return asInt64();  }
  inline uint64_t as<uint64_t>() const { return asUInt64(); }
  inline float as<float>() const       { return asFloat();  }
  inline double as<double>() const     { return asDouble(); }
  inline std::string as<std::string>() const { return asString(); }
  inline bool as<bool>() const         { return asBool(); }
*/

  /** Assign a value to a Value node in the tree.
   * If a previous operator[] found a match, this does a replace.
   * If a previous operator[] did not find a match in the tree
   * the current Value is a Zombie and not attached to the tree.
   * But in this case its requested key is remembered.  A subsequent
   * operator= will assign this node to the tree with the remembered key.
   * The parent will become a map if it is not already.
   */
  void operator=(char *val);
  void operator=(const std::string &val);
  void operator=(int8_t val);
  void operator=(int16_t val);
  void operator=(uint16_t val);
  void operator=(int32_t val);
  void operator=(uint32_t val);
  void operator=(int64_t val);
  void operator=(uint64_t val);
  void operator=(bool val);
  void operator=(float val);
  void operator=(double val);
  void operator=(std::vector<UInt32>);

  /**
   * Make a deep copy of the tree
   */
  Value copy() const {
    Value root;
    copy(&root);
    return root;
  }
  /**
   * Remove a node from the tree. (an element of map or sequence)
   * It is assumed that this is a low use function because
   * it will not be very efficent.
   */
  void remove();

  // compare two nodes in the tree.
  bool operator==(const Value &v) const;

  // return false if node is empty:   if (vm) { do something }
  explicit operator bool() const { return !isEmpty(); }

  // extract a Vector
  template <typename T> std::vector<T> asVector() const {
    std::vector<T> v;
    if (!isSequence())
      NTA_THROW << "Not a sequence node.";
    for (size_t i = 0; i < size(); i++) { // iterate through the children of this node.
      const Value &n = (*this)[i];
      try {
        if (n.isScalar()) {
          v.push_back(n.as<T>());
        }
      } catch (std::exception &e) {
        NTA_THROW << "Invalid vector element; " << e.what();
      }
    }
    return v;
  }

  // extract a map. Key is always a string.
  template <typename T> std::map<std::string, T> asMap() const {
    std::map<std::string, T> v;
    for (auto iter = cbegin(); iter != cend(); iter++) { // iterate through the children of this node.
      const Value &n = iter->second;
      try {
        if (n.isScalar()) {
          v[n.key_] = n.as<T>();
        } else {
          // non-scalar field.  Ignore
        }
      } catch (std::exception &e) {
        // probably bad conversion of scalar to requested type.
        NTA_THROW << "Invalid map element[\"" << n.key_ << "\"] " << e.what();
      }
    }
    return v;
  }

  // serializing routines
  std::string to_yaml() const;
  std::string to_json() const;

  // Access for backward compatability
  template <typename T> T getScalarT(const std::string &key) const { // throws
    return (*this)[key].as<T>();
  }
  template <typename T> T getScalarT(const std::string &key, T defaultValue) const { // with default
    if ((*this)[key].type_ != Value::Category::Scalar)
      return defaultValue;
    return (*this)[key].as<T>();
  }
  std::string getString(const std::string &key, const std::string &defaultValue) const {
    if ((*this)[key].type_ != Value::Category::Scalar)
      return defaultValue;
    return (*this)[key].str();
  }

  friend std::ostream &operator<<(std::ostream &f, const htm::Value &vm);

  std::map<std::string, Value>::iterator begin() { return map_.begin(); }
  std::map<std::string, Value>::iterator end() { return map_.end(); }
  std::map<std::string, Value>::const_iterator begin() const { return map_.begin(); }
  std::map<std::string, Value>::const_iterator end() const { return map_.end(); }
  std::map<std::string, Value>::const_iterator cbegin() const { return map_.cbegin(); }
  std::map<std::string, Value>::const_iterator cend() const { return map_.cend(); }

private:
  void assign(std::string val); // add a scalar
  void addToParent();           // Assign to the map/vector in the parent
  void copy(Value *target) const;
  void cleanup();

  int8_t asInt8() const;
  int16_t asInt16() const;
  uint16_t asUInt16() const;
  int32_t asInt32() const;
  uint32_t asUInt32() const;
  int64_t asInt64() const;
  uint64_t asUInt64() const;
  float asFloat() const;
  double asDouble() const;
  std::string asString() const;
  bool asBool() const ;


  enum Category type_; // type of node
  std::map<std::string, Value> map_;
  std::vector<std::map<std::string, Value>::iterator> vec_;
  Value *parent_;      // a pointer to the parent node  (do not delete)
  std::string scalar_; // scalar value.  Not used on Map or Sequence nodes
  std::string key_;    // The key for this node.
  size_t index_;       // initial index  Also,  ZOMBIE_MAP in zombie nodes.
  std::shared_ptr<Value> zombie_;      // pointer to a zombie child Value node. Returned when operator[] not found.
  Value *assigned_;   // For a zombie, a pointer to assigned node in tree.
};



using ValueMap = Value;



// A work-around for not allowing Explicit template specializations in the class scope.
// This is a bug in the GCC compiler before C++17, clang before C++14.  MSVC is ok.
template<> inline int8_t Value::as<int8_t>() const     { return asInt8();   }
template<> inline int16_t Value::as<int16_t>() const   { return asInt16();  }
template<> inline uint16_t Value::as<uint16_t>() const { return asUInt16(); }
template<> inline int32_t Value::as<int32_t>() const   { return asInt32();  }
template<> inline uint32_t Value::as<uint32_t>() const { return asUInt32(); }
template<> inline int64_t Value::as<int64_t>() const   { return asInt64();  }
template<> inline uint64_t Value::as<uint64_t>() const { return asUInt64(); }
template<> inline float Value::as<float>() const       { return asFloat();  }
template<> inline double Value::as<double>() const     { return asDouble(); }
template<> inline std::string Value::as<std::string>() const { return asString(); }
template<> inline bool Value::as<bool>() const         { return asBool(); }

} // namespace htm

#endif //  NTA_VALUE_HPP
