/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * Author: David Keeney, 10/2019
 *              dkeeney@gmail.com
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

#include <htm/ntypes/BasicType.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/utils/Log.hpp>

#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using namespace htm;

#define ZOMBIE_MAP ((size_t)-1) // Means the key of zombie was a map key
#define ZOMBIE_SEQ 0            // Means the key of zombie was a seq key

//////////////////////////////////////////////////////////////

////#ifdef YAML_PARSER_yamlcpp
#include <yaml-cpp/yaml.h>
static void setNode(Value &val, const YAML::Node &node);

// Parse YAML or JSON string document into the tree.
Value &Value::parse(const std::string &yaml_string) {
  // If this Value node is being re-used (like in unit tests)
  // we need to clear variables.
  vec_.clear();
  map_.clear();
  scalar_ = "";
  type_ = Value::Category::Empty;
  parent_ = nullptr;

  YAML::Node node = YAML::Load(yaml_string);
  // walk the tree and copy data into our structure

  setNode(*this, node);
  this->cleanup();
  return *this;
}

static void setNode(Value &val, const YAML::Node &node) {
  std::pair<std::map<std::string, Value>::iterator, bool> ret;
  if (node.IsScalar()) {
    val = node.as<std::string>();
  } else if (node.IsSequence()) {
    for (size_t i = 0; i < node.size(); i++) {
      setNode(val[i], node[i]);
    }
  } else if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); it++) {
      std::string key = it->first.as<std::string>();
      setNode(val[key], it->second);
    }
  }
}

////#endif // YAML_PARSER_yamlcpp

/////////////////////////////////////////////////////////////////////////////////////////

// Constructor
Value::Value() {
  type_ = Value::Category::Empty;
  parent_ = nullptr;
  zombie_ = nullptr;
  assigned_ = nullptr;
  index_ = ZOMBIE_MAP;
}


// checking content of a parameter
// enum ValueMap::Category { Empty = 0, Scalar, Sequence, Map };
ValueMap::Category Value::getCategory() const { return type_; }

bool Value::contains(const std::string& key) const { return (map_.find(key) != map_.end()); }

bool Value::isScalar() const { return type_ == Value::Category::Scalar; }
bool Value::isSequence() const { return type_ == Value::Category::Sequence; }
bool Value::isMap() const { return type_ == Value::Category::Map; }
bool Value::isEmpty() const { return type_ == Value::Category::Empty; }

size_t Value::size() const {
  NTA_CHECK(map_.size() == vec_.size()) << "Detected Corruption of ValueMap structure";
  return map_.size();
}

// Accessing members of a map
// If not found, a Zombie Value object is returned.
// An error will be displayed when you try to access the value in the Zombie Value object.
// If you assign something to the Zombie Value, it will insert it into the tree with the saved key.
Value& Value::operator[](const std::string& key) {
  if (assigned_)
    return (*assigned_)[key];
  auto it = map_.find(key);
  if (it == map_.end()) {
    // not found. Create a zombie in case we will later assign something to this key.
    //  Its type is Value::Category::Empty.
    // NOTE: only one zombie per parent can exist at a time.
    zombie_.reset(new Value());
    zombie_->parent_ = this;
    zombie_->scalar_ = key;
    zombie_->key_ = key;
    zombie_->index_ = ZOMBIE_MAP;
    return *zombie_;
  }
  else
    return it->second;
}
const Value& Value::operator[](const std::string& key) const {
  if (assigned_)
    return (*assigned_)[key];
  auto it = map_.find(key);
  if (it == map_.end()) {
    // not found. Create a (const) zombie which signals if found or not but
    // cannot be used with an assignment.  Its type is Value::Category::Empty.
    static Value const_zombie; // This is a constant
    return const_zombie;
  }
  else
    return it->second;
}

// accessing members of a sequence
Value& Value::operator[](size_t index) {
  if (assigned_)
    return (*assigned_)[index];
  if (index < vec_.size())
    return vec_[index]->second;
  else if (index == vec_.size()) {
    // Not found, create a zombie in case we later assign it.
    // Note that the index can ONLY be the size-of-vector.
    // Make sure the key is uneque. append '-'s until it is.
    std::string key = std::to_string(index);
    while (true) {
      if (map_.find(key) == map_.end())
        break;
      key += "-";
    }
    zombie_.reset(new Value());
    zombie_->parent_ = this;
    zombie_->key_ = key;
    zombie_->index_ = ZOMBIE_SEQ;
    return *zombie_;
  }
  NTA_THROW << "Index out of range; " << index;
}
const Value& Value::operator[](size_t index) const {
  if (assigned_)
    return (*assigned_)[index];
  if (index < vec_.size())
    return vec_[index]->second;
  NTA_THROW << "Index out of range; " << index; // is const so cannot make a zombie
}

std::string Value::str() const {
  NTA_CHECK(type_ == Value::Category::Scalar);
  return scalar_;
}
const char* Value::c_str() const {
  NTA_CHECK(type_ == Value::Category::Scalar);
  return scalar_.c_str();
}

std::string Value::key() const { return key_; }

std::vector<std::string> Value::getKeys() const {
  NTA_CHECK(isMap()) << "This is not a map.";
  std::vector<std::string> v;
  for (auto it = begin(); it != end(); it++) {
    v.push_back(it->first);
  }
  return v;
}

// Insert this node into the parent.
// Requires that there was a key (either string or index) unless it is root
// and if it was a string key, its index will be ZOMBIE_MAP.
void Value::addToParent() {
  std::pair<std::map<std::string, Value>::iterator, bool> ret;
  if (parent_ == nullptr)
    return; // This is the root
  NTA_CHECK(!key_.empty()) << "No key was provided.  Use node[key] = value.";
  if (parent_->type_ == Value::Category::Empty) {
    parent_->addToParent();
    if (parent_->assigned_)
      parent_ = parent_->assigned_;
  }
  bool map_key = (index_ == ZOMBIE_MAP);

  // Add the node to the parent.
  index_ = parent_->vec_.size();
  ret = parent_->map_.insert(std::pair<std::string, Value>(key_, *this));
  parent_->vec_.push_back(ret.first);
  assigned_ = &ret.first->second;

  NTA_CHECK(parent_->map_.size() == parent_->vec_.size()) << "Detected Corruption of ValueMap structure";

  if (map_key)
    parent_->type_ = Value::Category::Map;
  else if (parent_->type_ == Value::Category::Empty)
    parent_->type_ = Value::Category::Sequence;
}

void Value::assign(std::string val) {
  std::pair<std::map<std::string, Value>::iterator, bool> ret;
  if (type_ == Value::Category::Empty) { // previous search was false
    // This is a zombie node. By assigning a value we add it to the tree.
    // The key was already set in the operator[].

    // Add to parent.
    // If its parent is also a zombie, add it to the tree as well.
    addToParent();
    assigned_->scalar_ = val;
    assigned_->type_ = Value::Category::Scalar;
  } else {
    // Must be a value already in the tree.  Do a replace.
    if (type_ != Value::Category::Scalar) {
      map_.clear();
      vec_.clear();
      type_ = Value::Category::Scalar;
    }
    scalar_ = val;
  }
}
// Assign a value converted from a specified type T.
void Value::operator=(char *val) { assign(val); }
void Value::operator=(const std::string &val) { assign(val); }
void Value::operator=(int8_t val) { assign(std::to_string(val)); }
void Value::operator=(int16_t val) { assign(std::to_string(val)); }
void Value::operator=(uint16_t val) { assign(std::to_string(val)); }
void Value::operator=(int32_t val) { assign(std::to_string(val)); }
void Value::operator=(uint32_t val) { assign(std::to_string(val)); }
void Value::operator=(int64_t val) { assign(std::to_string(val)); }
void Value::operator=(uint64_t val) { assign(std::to_string(val)); }
void Value::operator=(bool val) { assign((val) ? "true" : "false"); }
void Value::operator=(float val) { assign(std::to_string(val)); }
void Value::operator=(double val) { assign(std::to_string(val)); }
void Value::operator=(std::vector<UInt32> val) {
  // Insert the contents of the vector into this node.
  map_.clear();
  vec_.clear();
  for (size_t i = 0; i < val.size(); i++) {
    operator[](i) = std::to_string(val[i]);
  }
}

void Value::copy(Value *target) const {
  if (assigned_)
    assigned_->copy(target);

  target->type_ = type_;
  target->scalar_ = scalar_;
  target->key_ = key_;
  target->index_ = index_;

  std::pair<std::map<std::string, Value>::iterator, bool> ret;
  for (size_t i = 0; i < vec_.size(); i++) {
    std::string key = vec_[i]->first;
    Value itm;
    ret = target->map_.insert(std::pair<std::string, Value>(key, itm));
    target->vec_.push_back(ret.first);
    vec_[i]->second.copy(&ret.first->second);
    vec_[i]->second.parent_ = target;
  }
}

void Value::cleanup() {
  if (isEmpty())
    return;

  zombie_.reset();
  if (isScalar()) return;
  for (size_t i = 0; i < vec_.size(); i++)
      vec_[i]->second.zombie_.reset();
}


void Value::remove() {
  Value *node = (assigned_) ? assigned_ : this;
  NTA_CHECK(!node->isEmpty()) << "Item not found."; // current node is a zombie.
  if (node->parent_ == nullptr) {
    // This is root.  Just clear the map.
    node->vec_.clear();
    node->map_.clear();
    node->type_ = Value::Category::Empty;
    return;
  }
  NTA_CHECK(node->parent_->vec_[index_]->second == *node);
  if (node->parent_->vec_.size() == 1) {
    // Last node in parent, remove parent.
    node->parent_->remove();
    return;
  }
  std::string key = node->parent_->vec_[index_]->first;
  auto itr = node->parent_->map_.find(key);

  // adjust the index on all following items.
  // We have to do it here because as soon as we erase the map item
  // it will delete 'this'.
  for (size_t i = index_+1; i < node->parent_->vec_.size(); i++) {
    node->parent_->vec_[i]->second.index_ = i-1;
  }

  node->parent_->vec_.erase(node->parent_->vec_.begin() + index_);
  node->parent_->map_.erase(itr);
  // The node object is deleted. Do no try to access it.
}

// Compare two nodes recursively to see if content is same.
static bool equals(const Value &n1, const Value &n2) {
  if (n1.getCategory() != n2.getCategory())
    return false;
  if (n1.isSequence()) {
    if (n1.size() != n2.size())
      return false;
    for (size_t i = 0; i < n1.size(); i++)
      if (!equals(n1[i], n2[i]))
        return false;
    return true;
  }
  if (n1.isMap()) {
    if (n1.size() != n2.size())
      return false;
    for (auto it : n1) {
      if (!n2[it.first])
        return false;
      if (!equals(it.second, n2[it.first]))
        return false;
    }
    return true;
  }
  if (n1.isScalar()) {
    if (n1.str() == n2.str())
      return true;
  }
  return false;
}
bool Value::operator==(const Value &v) const { return equals(*this, v); }

// Explicit implementations for as<T>()
#define NTA_CONVERT(T, I)                                                                                              \
  if (type_ != Value::Category::Scalar)                                                                                \
    NTA_THROW << "value not found.";                                                                                   \
  errno = 0;                                                                                                           \
  char *end;                                                                                                           \
  T val = (T)I;                                                                                                        \
  if (errno)                                                                                                           \
    NTA_THROW << "In '" << scalar_ << "' numeric conversion error: " << std::strerror(errno);                          \
  if (*end != '\0')                                                                                                    \
    NTA_THROW << "In '" << scalar_ << "' numeric conversion error: invalid char.";                                     \
  return val;

  int8_t Value::asInt8() const { NTA_CONVERT(int8_t, std::strtol(scalar_.c_str(), &end, 0)); }
  int16_t Value::asInt16() const { NTA_CONVERT(int16_t, std::strtol(scalar_.c_str(), &end, 0)); }
  uint16_t Value::asUInt16() const { NTA_CONVERT(uint16_t, std::strtoul(scalar_.c_str(), &end, 0)); }
  int32_t Value::asInt32() const { NTA_CONVERT(int32_t, std::strtol(scalar_.c_str(), &end, 0)); }
  uint32_t Value::asUInt32() const { NTA_CONVERT(uint32_t, std::strtoul(scalar_.c_str(), &end, 0)); }
  int64_t Value::asInt64() const { NTA_CONVERT(int64_t, std::strtoll(scalar_.c_str(), &end, 0)); }
  uint64_t Value::asUInt64() const { NTA_CONVERT(uint64_t, std::strtoull(scalar_.c_str(), &end, 0)); }
  float Value::asFloat() const { NTA_CONVERT(float, std::strtof(scalar_.c_str(), &end)); }
  double Value::asDouble() const { NTA_CONVERT(double, std::strtod(scalar_.c_str(), &end)); }
  std::string Value::asString() const {
    if (type_ != Value::Category::Scalar)
      NTA_THROW << "value not found.";
    return scalar_;
  }
  bool Value::asBool() const {
    if (type_ != Value::Category::Scalar)
      NTA_THROW << "value not found. " << scalar_;
    std::string val = str();
    transform(val.begin(), val.end(), val.begin(), ::tolower);
    if (val == "true" || val == "on" || val == "1")
      return true;
    if (val == "false" || val == "off" || val == "0")
      return false;
    NTA_THROW << "Invalid value for a boolean. " << val;
  }



/**
 * a local function to apply escapes for a JSON string.
 */
static void escape_json(std::ostream &o, const std::string &s) {
  for (auto c = s.cbegin(); c != s.cend(); c++) {
    switch (*c) {
    case '"':
      o << "\\\"";
      break;
    case '\\':
      o << "\\\\";
      break;
    case '\b':
      o << "\\b";
      break;
    case '\f':
      o << "\\f";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    case '\t':
      o << "\\t";
      break;
    default:
      if ('\x00' <= *c && *c <= '\x1f') {
        o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)*c;
      } else {
        o << *c;
      }
    }
  }
}

static void to_json(std::ostream &f, const htm::Value &v) {
  bool first = true;
  std::string s;
  switch (v.getCategory()) {
  case Value::Empty:
    return;
  case Value::Scalar:
    s = v.str();
    if (std::regex_match(s, std::regex("^[-+]?[0-9]+([.][0-9]+)?$"))) {
      escape_json(f, s);
    } else {
      f << '"';
      escape_json(f, s);
      f << '"';
    }
    break;
  case Value::Sequence:
    f << "[";
    for (size_t i = 0; i < v.size(); i++) {
      if (!first)
        f << ", ";
      first = false;
      const Value &n = v[i];
      to_json(f, n);
    }
    f << "]";
    break;
  case Value::Map:
    f << "{";
    for (size_t i = 0; i < v.size(); i++) {
      if (!first)
        f << ", ";
      first = false;
      const Value &n = v[i];
      f << n.key() << ": ";
      to_json(f, n);
    }
    f << "}";
    break;
  }
}

std::string Value::to_json() const {
  std::stringstream f;
  ::to_json(f, *this);
  return f.str();
}

static void escape_yaml(std::ostream &o, const std::string &s, const std::string &indent) {
  if (std::strchr(s.c_str(), '\n')) {
    // contains newlines
    o << " |";  // all blanks are significant
    const char *from = s.c_str();
    const char *to = from;
    while ((to = std::strchr(to, '\n')) != NULL) {
      std::string line(from, to);
      o << "\n" + indent + line;
      ++to;
      from = to;
    }
    o << "\n" + indent + from;
  } else {
    o << s;
  }
}

static void to_yaml(std::ostream & f, const htm::Value &v, std::string indent) {
  bool first = true;
  std::string s;
  switch (v.getCategory()) {
  case Value::Empty:
    return;
  case Value::Scalar:
    s = v.str();
    escape_yaml(f, s, indent);
    f << "\n";
    break;
  case Value::Sequence:
    for (size_t i = 0; i < v.size(); i++) {
      const Value &n = v[i];
      f << indent << "- ";
      if (n.isMap() || n.isSequence())
        f << "\n";
      to_yaml(f, n, indent + "  ");
    }
    break;
  case Value::Map:
    for (size_t i = 0; i < v.size(); i++) {
      const Value &n = v[i];
      f << indent << n.key() << ": ";
      if (n.isMap() || n.isSequence())
        f << "\n";
      to_yaml(f, n, indent + "  ");
    }
    break;
  }
}

std::string Value::to_yaml() const {
  std::stringstream f;
  ::to_yaml(f, *this, "");
  return f.str();
}

namespace htm {
std::ostream &operator<<(std::ostream &f, const htm::Value &v) {
  f << v.to_json();
  return f;
}
} // namespace htm
