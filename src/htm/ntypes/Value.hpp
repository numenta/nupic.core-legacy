/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * author: David Keeney dkeeney@gmail.com, 10/2019
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

/***********************************************
|  Value object - a container for data structures that have been parsed
|  from a YAML or JSON string.  It can hold any structure that can be
|  parsed from these but does not implement Alias or reference features.
|  YAML is a superset of JSON format but we only use the JSON compatible features.
|
|  Example:
|   Consider the JSON string:  {scalar: 456, array: [1, 2, 3, 4], string: "true"}
|   Written as YAML it is:
|          scalar: 456
|          array:
|            - 1
|            - 2
|            - 3
|            - 4
|          string: "true"
|
|   When parsed this is a Map containing three elements indexed as "scalar", "array", and "string".
|   - The "scalar" element contains the number 456.
|   - The "array" element contains a Sequence object which holds 4 elements which happen to be numbers.
|   - The "string" element contains string with the value "true".
|
|         std::string source = "{scalar: 456, array: [1, 2, 3, 4], string: \"true\"}";
|         Value v;
|         v.parse(source);
|
|         v["scalar"].as<int>      --will return 456 as an integer
|         v["array"][1].as<int>    --will return 2 as an integer
|         v["string"].str()        --will return "true" as a string.
|         v["string"].as<bool>()   --will return True as a boolean.
|
|  Usage: Parsing
|     Value &parse(const std::string &yaml_string);
|        Parses the yaml_string as a tree with the current object ('this') as the root.
|        Returns a reference to 'this' so you can do chaining of calls.  The tree consist
|        of Value objects that are linked together.  The root is also a Value object.
|        A Value object can be a Scalar, a Sequence, or a Map.  It can also be Empty.
|
|  Usage: Access
|     Value& v[key]
|        Indexes into the Value node with the given key and returns a reference to the selected
|        item similar to the way STL objects can be accessed.
|        - If the base Value (v) is a Scalar, this will give an error.
|
|        - The 'key' can be a string; in which case it does a key lookup in that base Value node 
|          as if it were a std::map<string, Value>. If found, the returned reference is to 
|          the Value object corresponding to that key which can be a Scalar, a Sequence, or a Map.
|          If not found it returns a reference to a zombie Value node which has an isEmpty() 
|          attribute. Unlike STL, it is not added to the tree until it is assigned a value.
|
|        - The 'key' can also be an integer in which case  it indexes as if it were 
|          std::vector<Value>. If the numeric index is within range the Value node 
|          returned is the Value at that position in the sequence.  This can be a Scalar, 
|          a Sequence, or a Map. An index of v.size() will return a reference to a zombie 
|          Value node as with the string key. All other out-of-range numeric indexes will 
|          give an error. If the base Value node is a Map type, the values returned will 
|          be in the order the values were added so this allows Map Values to be accessed 
|          by numeric indexes.
|
|        - keys can be stacked with conversions... for example:  
|                 T val = v[key1][key2].as<T>();
|
|     bool contains(const std::string& key)
|        Returns true if the key matches something in the base Value node.
|
|     std::vector<std::string> getKeys()
|        Returns a vector containing all keys in the base Value node.
|
|     size_t size()
|         Returns the number of elements in the base Value node.
|
|     bool operator==()
|         Deep compare of two Value objects for attributes and content being the same.
|
|  Usage:  Attributes
|      Value::Category getCategory()
|          Returns the type of Value object.
|      Value objects can have an attribute or type of:
|        - Value::Category::Scalar   - holds a value; a leaf on the tree. All values are 
|                                      strings.  v.isScaler() is true.
|        - Value::Category::Sequence - An array of Value nodes. v.isSequence() is true;
|        - Value::Category::Map      - holds a string key plus a Value node. 
|                                      v.isMap() is true;
|        - Value::Category::Empty    - A Value node that has not been assigned a value. 
|                                     (i.e. it is a zombie node).  v.isEmpty() is true.
|
|  Usage: Conversions
|     std::string str()
|     char* c_str()
|         Returns the raw string value.   Same as v.as<std::string>() but faster.
|         The base Value (v) must be a Scaler or it will throw an exception.
|
|     T value = v.as<T>()
|         Converts the internal string value into the specified type T.  
|         T can be any numeric, boolean or an std::string.  It can also be
|         anything that std::stringstream can convert.
|         The base Value (v) must be a Scaler or it will throw an exception.
|
|     std::vector<T> v.asVector<T>()
|     std::map<std::string, T>  v.asMap<T>()
|         Converts the scalar items in a base Value into a vector or a map.
|         The base Value (v) must be a Sequence or a Map or it will throw an exception.
|         Any element that is not a Scalar will be skipped.
|
|  Usage: Iterators
|     Iterations are similar to STL objects.
|     The iterated Value node must be a Sequence or a Map or it throws an exception.
|
|     By range:
|         for (auto itm : vm) {
|            std::string key = itm.first;
|            Value& v = itm.second;
|         }
|
|     By iterator:     const and non-const begin() and cbegin() supported.
|         for (auto itr = vm.begin(); itr != vm.end(); itr++) {
|            std::string key = itr->first;
|            Value& v = itr->second;
|         }
|
|     By index:
|         for (size_t i = 0; i < vm.size(); i++) {
|            Value& v = vm[i];
|         }
|
|  Usage: Serialization
|     std::string to_yaml();
|         Returns a yaml formatted string from the contents of the tree.
|
|     std::string to_json();
|         Returns a JSON formatted string from the contents of the tree.
|
|     operator<<(std::ostream &f, const htm::Value &vm);
|         Returns a JSON formatted string from the contents of the tree.
|
|  Usage: Direct Modification
|     Any Value node can be assigned to.  If the node was a zombie the node 
|     is added to the tree. A zombie node can be the result of a new allocation 
|     or a failed lookup i.e. v[key].   If the node was already in the tree the 
|     assignment changes its value and it becomes a Scalar. If the node had 
|     previously been a Map or Sequence, all subordinate nodes  are released.  
|     The right hand side of the expression can be any type of number, a bool, 
|     a string, or a vector<UInt32>.
|            v = "abc";            -- assigns a string value to the node.
|            v = 25.6f;            -- converts to a string and assigns to the node.
|            v[2]["name"][0] = 6;  -- assigns, creating maps and sequences as needed.
|
|     void remove()
|          Will remove the base Value node and all subordinate nodes.
|          Indexes of Sequences are adjusted.  References to Values in the tree become invalid.
|
|     Value copy()
|          Performs a deep copy of that Value node and below. The returned node
|          is the root of the new tree.
|
|  Usage:  Backward Compatability
|     ValueMap is the same as a Value object.
|
|     std::string getString(const std::string &key, const std::string &defaultValue);
|        Return the raw string value coresponding to the key.  Same as str() with a
|        check to see if it exists.
|
|     T getScalarT(const std::string &key);
|     T getScalarT(const std::string &key, T defaultValue);
|        Return a converted value from the Value object corresponding to the key.
|        If the default value is not provided it throws an exception when the key is not found.
|        These are the same as:  T value = v[key].as<T>();
|
************************************************/
#ifndef NTA_VALUE_HPP
#define NTA_VALUE_HPP

#include <htm/utils/Log.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <thread>

namespace htm {



class Value {
public:
  enum Category { Empty = 0, Scalar, Sequence, Map };

private:
  // All internal data held by the Value node is contained in this structure
  // within a shared_ptr.  When a Zombie node is assigned a value, this data
  // is copied to the real node as it is inserted into the map in the parent.
  // So a reference to the Zombie or the real node will access the same data.
  struct internals {
    enum Value::Category type_;                // type of node, Map, Sequence, Scalar, Empty
    std::map<std::string, Value> map_;         // values in a Map or Sequence
    std::vector<std::map<std::string, Value>::iterator> vec_;
                                               // index of iterators pointing to Map entries.
    std::shared_ptr<struct internals> parent_; // a pointer to the parent's core 
    std::string scalar_;                       // scalar value.  Not used on Map or Sequence nodes
    std::string key_;                          // The key for this node.
    size_t index_;                             // initial index or ZOMBIE_MAP/ZOMBIE_SEQ in zombie nodes.
    std::map<std::thread::id, std::shared_ptr<Value>> zombie_; 
                                               // map of pointers to zombie child Value nodes.
                                               // One element for each thread accessing the class.
                                               // Returned by operator[] when it does not found a key.
    void addToParent(Value& node);             // Assign to the map/vector in the parent
    void remove();                             // Remove this node from parent
  };
  std::shared_ptr<struct internals> core_;

public:
  Value();

  // Parse a Yaml or JSON string an assign it to this node in the tree.
  Value &parse(const std::string &yaml_string);

  bool contains(const std::string &key) const;
  size_t size() const;

  // type of value in the Value node
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
    if (core_->type_ != Category::Scalar)
      NTA_THROW << "not found. '" << core_->scalar_ << "'";
    T result;
    std::stringstream ss(core_->scalar_);
    ss >> result;
    return result;
  }

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
          v[n.core_->key_] = n.as<T>();
        } else {
          // non-scalar field.  Ignore
        }
      } catch (std::exception &e) {
        // probably bad conversion of scalar to requested type.
        NTA_THROW << "Invalid map element[\"" << n.core_->key_ << "\"] " << e.what();
      }
    }
    return v;
  }

  // serializing routines
  std::string to_yaml() const;
  std::string to_json() const;

  // Access for backward compatability
  template <typename T> T getScalarT(const std::string &key) const { // throws
    try {
      return (*this)[key].as<T>();
    } catch (Exception &e) {
      NTA_THROW << "ValueMap.getScalarT(\"" << key << "\") - " << e.what();
    }
  }
  template <typename T> T getScalarT(const std::string &key, T defaultValue) const { // with default
    if ((*this)[key].core_->type_ != Value::Category::Scalar)
      return defaultValue;
    try {
      return (*this)[key].as<T>();
    } catch (Exception &e) {
      NTA_THROW << "ValueMap.getScalarT(\"" << key << "\") - " << e.what();
    }
  }
  std::string getString(const std::string &key, const std::string &defaultValue) const {
    if ((*this)[key].core_->type_ != Value::Category::Scalar)
      return defaultValue;
    return (*this)[key].str();
  }

  friend std::ostream &operator<<(std::ostream &f, const htm::Value &vm);

  std::map<std::string, Value>::iterator begin() { return core_->map_.begin(); }
  std::map<std::string, Value>::iterator end() { return core_->map_.end(); }
  std::map<std::string, Value>::const_iterator begin() const { return core_->map_.begin(); }
  std::map<std::string, Value>::const_iterator end() const { return core_->map_.end(); }
  std::map<std::string, Value>::const_iterator cbegin() const { return core_->map_.cbegin(); }
  std::map<std::string, Value>::const_iterator cend() const { return core_->map_.cend(); }

  // for validating linkages within tree -- for debugging only.
  bool check(const std::string &indent = "") const;

private:
  void assign(std::string val); // add a scalar
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
