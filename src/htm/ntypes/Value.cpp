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

#include <algorithm> // transform
#include <cerrno>
#include <cstring> // std::strerror(errno)
#include <iomanip>
#include <iostream>
#include <regex>
#include <stack>
#include <mutex>

static std::mutex g_tree_mutex;

using namespace htm;

#define ZOMBIE_MAP ((size_t)-1) // Means the key of zombie was a map key
#define ZOMBIE_SEQ 0            // Means the key of zombie was a seq key

//////////////////////////////////////////////////////////////
// Parser interface.
// Place code to interface with a yaml parser here.
// Note: #define YAML_PARSERxxxx is set by the 'external' module that loaded the parser.
//           Only one parser is loaded.  See external/CMakeLists.txt

// Within the parse() function the interface should parse the yaml_string and then
// populate the resulting tree under the Value root (the 'this' object).
// The parse function should return a reference to 'this' so additional
// functions can be chained.
//
// As a result of the parse, the root Value node may be a Scalar, Sequence, or a Map.

#ifdef YAML_PARSER_libYaml
// All interface for the libYaml parser must be encapulated in this section.
// Documentation:
//   https://pyyaml.org/wiki/LibYAML
//   http://staskobzar.blogspot.com/2017/04/yaml-documents-parsing-with-libyaml-in-c.html
//   https://www.wpsoftware.net/andrew/pages/libyaml.html
#define YAML_DECLARE_STATIC
#include <yaml.h>

// Set verbose to true if you need to debug your yaml string.
// somethings this is the only way to unscriable a syntax problem.
#define VERBOSE  if (verbose) std::cerr << "[          ] "
static bool verbose = false; 

// Parse YAML or JSON string document into the tree root.
Value &Value::parse(const std::string &yaml_string) {
  std::stack<Value *> stack; // top of stack is parent.
  // We need to clear variables, just in case it previously had a value.
  core_->vec_.clear();
  core_->map_.clear();
  core_->scalar_ = "";
  core_->type_ = Value::Category::Empty;
  //core_->parent_.reset();

  Value *node = this;

  yaml_parser_t parser;
  yaml_event_t event;
  enum state_t { start_state = 0, seq_state, map_state, map_key };
  state_t  state = state_t::start_state;
  yaml_event_type_e event_type = YAML_NO_EVENT;
  std::string key;

  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser, (const unsigned char *)yaml_string.c_str(), yaml_string.size());

  do {
    int status;
    try {
      status = yaml_parser_parse(&parser, &event);
    } catch (std::exception& e) {
      std::string err = "Parse Error, Exception in yaml parser: " + std::string(e.what());
      yaml_parser_delete(&parser);
      NTA_THROW << err;
    } catch (...) {
      std::string err = "Parse Error, Unknown Exception in yaml parser.";
      yaml_parser_delete(&parser);
      NTA_THROW << err;
    }

    if (!status) {
      std::string err = "Parse Error " + std::to_string(parser.error) + ": " + std::string(parser.problem) +
                        ", offset: " + std::to_string(parser.problem_offset) +
                        ", context: " + std::string(parser.context);
      if (!key.empty())
        err += " following key: `" + key + "'.";
      yaml_parser_delete(&parser);
      VERBOSE << err << std::endl;
      NTA_THROW << err;
    }
    event_type = event.type;
    VERBOSE << "Event: " << event_type << std::endl;
    try {
      switch (event_type) {
      case YAML_NO_EVENT: break;
      case YAML_STREAM_START_EVENT: break;
      case YAML_STREAM_END_EVENT: break;
      case YAML_DOCUMENT_START_EVENT:  state = start_state; break;
      case YAML_DOCUMENT_END_EVENT:  NTA_CHECK(node->isRoot()); break;
      case YAML_ALIAS_EVENT: break;
      case YAML_MAPPING_START_EVENT:
      case YAML_SEQUENCE_START_EVENT:
        switch (state) {
        case start_state:
          break;
        case seq_state:
          stack.push(node);
          node = &node->operator[](node->size());
          break;
        case map_key:
          stack.push(node);
          node = &node->operator[](key);
          break;
        default:
          break;
        }
        state = (event_type == YAML_SEQUENCE_START_EVENT)?seq_state : map_state;
        break;
      case YAML_MAPPING_END_EVENT:
      case YAML_SEQUENCE_END_EVENT:
        if (stack.size() > 0) {
          node = stack.top();
          stack.pop();
          state = (node->isSequence()) ? seq_state : (node->isMap()) ? map_state : start_state;
        } else {
          state = start_state;
        }
        break;
      // Data
      case YAML_SCALAR_EVENT: {
        std::string val((char *)event.data.scalar.value, event.data.scalar.length);
        switch (state) {
        case map_state:
          key = val;
          VERBOSE << "key: " << key << std::endl;
          state = map_key;
          break;
        case map_key:
          VERBOSE << "map Scalar value: " << val << std::endl;
          (*node)[key] = val;
          state = map_state;
          break;
        case seq_state:
          VERBOSE << "Seq Scalar value: " << val << std::endl;
          (*node)[node->size()] = val;
          state = seq_state;
          break;
        default:
          VERBOSE << "Scalar value: " << val << std::endl;
          (*node) = val;
          break;
        }
      } break;
      default:
        break;
      }
    } catch (const Exception &e) {
      yaml_event_delete(&event);
      yaml_parser_delete(&parser);
      NTA_THROW << "Parser error " << e.what();
    } catch (...) {
      yaml_event_delete(&event);
      yaml_parser_delete(&parser);
      NTA_THROW << "Parser error: unknown exception. ";
    }

    yaml_event_delete(&event);
  } while (event_type != YAML_STREAM_END_EVENT);
  yaml_parser_delete(&parser);

  NTA_CHECK(stack.empty()) << "Parsing syntax error. check your brackets.";

  this->cleanup();
  return *this;
}

/////////////////////////////////////////////////////////////////////////////////////////
#else // YAML_PARSER_yamlcpp

// All interface for yaml-cpp parser must be encapulated in this section.
#include <yaml-cpp/yaml.h>
static void setNode(Value &val, const YAML::Node &node);

// Parse YAML or JSON string document into the tree.
Value &Value::parse(const std::string &yaml_string) {
  if (assigned_)
    return assigned_->parse(yaml_string);
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

// Copy a yaml-cpp node to a Value node recursively.
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

#endif // YAML_PARSER_yamlcpp
/////////////////////////////////////////////////////////////////////////////////////////


// The remaining code is independent of the parser


// Constructor
Value::Value() {
  core_ = std::make_shared<struct internals>();
  core_->type_ = Value::Category::Empty;
  core_->index_ = ZOMBIE_MAP;
}

// checking content of a parameter
// enum ValueMap::Category { Empty = 0, Scalar, Sequence, Map };
ValueMap::Category Value::getCategory() const {
  return core_->type_;
}

bool Value::contains(const std::string &key) const {
  return (core_->map_.find(key) != core_->map_.end());
}

bool Value::isScalar() const { return getCategory() == Value::Category::Scalar; }
bool Value::isSequence() const { return getCategory() == Value::Category::Sequence; }
bool Value::isMap() const { return getCategory() == Value::Category::Map; }
bool Value::isEmpty() const { return getCategory() == Value::Category::Empty; }
bool Value::isRoot() const { return (!core_->parent_); }

size_t Value::size() const {
  NTA_CHECK(core_->map_.size() == core_->vec_.size()) << "Detected Corruption of ValueMap structure";
  return core_->map_.size();
}

// Accessing members of a map
// If not found, a Zombie Value object is returned.
// An error will be displayed when you try to access the value in the Zombie Value object.
// If you assign something to the Zombie Value, it will insert it into the tree with the saved key.
//
// What is this zombie?
// When we do the lookup with something like vm["hello"] and it finds something it returns 
// a reference to a node in the tree that matches the key. However, if the lookup did not 
// find anything we don't have anything to return a reference to.  What STL does is create 
// an entry in the tree and return a reference to that but I would rather not change the 
// tree until there is a value being assigned to it.  So what I do is create a zombie node, 
// assign it a key value and a pointer to its parent, and return a reference to it. This is 
// just a Value node of type Empty which is not linked into the tree but holds the key and
// a pointer to the parent in  case it is assigned to.
//
// When the operator= makes the assignment, it puts the value into the zombie, and then adds 
// the zombie to the map in the parent in the tree using the key, changing its type to Scalar.
// At that point it is no longer a zombie.
//
// The problem is where does that node exist until a value can be assigned? The caller has 
// only a reference to it. So what I do is put the zombie object on the parent node in a 
// map of shared_ptrs so I can make it thread safe. The down side is that there can be 
// only one outstanding lookup (or zombie) per parent node (per thread) that can be assigned to.
// Its scope lasts until the next lookup on that node within a thread.

Value &Value::operator[](const std::string &key) {
  auto it = core_->map_.find(key);
  if (it == core_->map_.end()) {
    // not found. Create a zombie in case we will later assign something to this key.
    // Its type is Value::Category::Empty. It holds the key and pointer to parent.
    // NOTE: only one zombie per parent per thread can exist at a time.
    //       Its scope is until the next call to operator[].

    // create the zombie for this thread.
    std::thread::id this_id = std::this_thread::get_id();
    std::shared_ptr<Value> zombie = std::make_shared<Value>();
    core_->zombie_[this_id] = zombie;
    zombie->core_->parent_ = this->core_;
    zombie->core_->key_ = key;
    zombie->core_->index_ = ZOMBIE_MAP;
    return (*zombie);
  } else
    return it->second;
}
const Value &Value::operator[](const std::string &key) const {
  auto it = core_->map_.find(key);
  if (it == core_->map_.end()) {
    // not found. Create a (const) zombie which signals if found or not but
    // cannot be used with an assignment.  Its type is Value::Category::Empty.
    static Value const_zombie; // This is a constant
    return const_zombie;
  } else
    return it->second;
}

// accessing members of a sequence
Value &Value::operator[](size_t index) {
  if (index < core_->vec_.size())
    return core_->vec_[index]->second;
  else if (index == core_->vec_.size()) {
    // Not found, create a zombie in case we later assign it.
    // Note that the index can ONLY be the size-of-vector.
    // Make sure the key is uneque. append '-'s until it is.
    std::string key = std::to_string(index);
    while (true) {
      if (core_->map_.find(key) == core_->map_.end())
        break;
      key += "-";
    }

    // Create the zombie for this thread
    std::thread::id this_id = std::this_thread::get_id();
    std::shared_ptr<Value> zombie = std::make_shared<Value>();
    core_->zombie_[this_id] = zombie;
    zombie->core_->parent_ = this->core_;
    zombie->core_->key_ = key;
    zombie->core_->index_ = ZOMBIE_SEQ;
    return (*zombie);
  }
  NTA_THROW << "Index out of range; " << index;
}
const Value &Value::operator[](size_t index) const {
  if (index < core_->vec_.size())
    return core_->vec_[index]->second;
  NTA_THROW << "Index out of range; " << index; // is const so cannot make a zombie
}

std::string Value::str() const {
  NTA_CHECK(core_->type_ == Value::Category::Scalar);
  return core_->scalar_;
}
const char *Value::c_str() const {
  NTA_CHECK(core_->type_ == Value::Category::Scalar);
  return core_->scalar_.c_str();
}

std::string Value::key() const { return core_->key_; }

std::vector<std::string> Value::getKeys() const {
  NTA_CHECK(core_->type_ == Value::Category::Map) << "This is not a map.";
  std::vector<std::string> v;
  for (auto it = begin(); it != end(); it++) {
    v.push_back(it->first);
  }
  return v;
}

// Insert this node into the parent.
// Requires that there was a key (either string or index) unless it is root
// and if it was a string key, its index will be ZOMBIE_MAP.
void Value::internals::addToParent(Value& node) {
  if (!parent_) {
    return; // This is root
  }
  NTA_CHECK(!key_.empty()) << "No key was provided.  Use node[key] = value.";
  // Make sure the parent is in the tree.
  if (parent_->type_ == Value::Category::Empty) {
    Value pnode;
    pnode.core_ = parent_;
    parent_->addToParent(pnode);
  }
  bool isKeyForMap = (index_ == ZOMBIE_MAP);

  // Add the node to the parent.
  g_tree_mutex.lock();
  try {
    auto itr = parent_->map_.find(key_);
    if (itr == parent_->map_.end()) {
      index_ = parent_->vec_.size();
      auto ret = parent_->map_.insert(std::pair<std::string, Value>(key_, node));
      parent_->vec_.push_back(ret.first);
    }
  } catch (std::exception &e) {
    g_tree_mutex.unlock();
    NTA_THROW << e.what();
  }
  g_tree_mutex.unlock();

  NTA_CHECK(parent_->map_.size() == parent_->vec_.size())
      << "Detected Corruption of ValueMap structure";

  // determine the type on the parent
  if (isKeyForMap)
    parent_->type_ = Value::Category::Map;
  else if (parent_->type_ == Value::Category::Empty)
    parent_->type_ = Value::Category::Sequence;
}

// Assign a Scalar value to a Value node.
void Value::assign(std::string val) {
  if (isRoot()) {
    core_->map_.clear();
    core_->vec_.clear();
    core_->scalar_ = val;
    core_->type_ = Value::Category::Scalar;
    return;
  }

  if (core_->type_ == Value::Category::Empty) { // previous search was false
    // This is a zombie node. By assigning a value we add it to the tree.
    // The key was already set in the operator[].

    // Add to parent.
    // If its parent is also a zombie, add it to the tree as well.
    core_->addToParent(*this);
    core_->scalar_ = val;
    core_->type_ = Value::Category::Scalar;
  } else {
    // Not a zombie
    // Must be a value already in the tree.  Do a replace.
    if (core_->type_ != Value::Category::Scalar) {
      core_->map_.clear();
      core_->vec_.clear();
      core_->type_ = Value::Category::Scalar;
    }
    core_->scalar_ = val;
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
  core_->map_.clear();
  core_->vec_.clear();
  for (size_t i = 0; i < val.size(); i++) {
    operator[](i) = std::to_string(val[i]);
  }
}

void Value::copy(Value *target) const {
  target->core_->type_ = core_->type_;
  target->core_->scalar_ = core_->scalar_;
  target->core_->key_ = core_->key_;
  target->core_->index_ = core_->index_;

  for (size_t i = 0; i < core_->vec_.size(); i++) {
    std::pair<std::map<std::string, Value>::iterator, bool> ret;
    std::string key = core_->vec_[i]->first;
    Value child_target; 
    child_target.core_->parent_ = target->core_;
    g_tree_mutex.lock();
    try {
      auto itr = target->core_->map_.find(key);
      if (itr == target->core_->map_.end()) {
        ret = target->core_->map_.insert(std::pair<std::string, Value>(key, child_target));
        target->core_->vec_.push_back(ret.first);
      }
    } catch (std::exception &e) {
      g_tree_mutex.unlock();
      NTA_THROW << e.what();
    }
    g_tree_mutex.unlock();
    core_->vec_[i]->second.copy(&ret.first->second);
  }
}

void Value::cleanup() {
  if (isEmpty())
    return;

  core_->zombie_.clear();
  if (isScalar())
    return;
  for (size_t i = 0; i < core_->vec_.size(); i++)
    core_->vec_[i]->second.core_->zombie_.clear();
}

void Value::remove() { core_->remove(); }

void Value::internals::remove() {
  if (!parent_) {
    // This is root.  Just clear the map.
    vec_.clear();
    map_.clear();
    type_ = Value::Category::Empty;
    return;
  }
  NTA_CHECK(type_ != Value::Category::Empty) << "Item not found."; // current node is a zombie not assigned.
  if (parent_->vec_.size() == 1) {
    // Last node in parent, remove parent.
    parent_->remove();
    return;
  }
  std::string key = parent_->vec_[index_]->first;
  auto itr = parent_->map_.find(key);

  // adjust the index on all following items.
  // We have to do it here because as soon as we erase the map item
  // it will delete 'this'.
  for (size_t i = index_ + 1; i < parent_->vec_.size(); i++) {
    parent_->vec_[i]->second.core_->index_ = i - 1;
  }

  parent_->vec_.erase(parent_->vec_.begin() + index_);
  parent_->map_.erase(itr);
  // The 'this' object is deleted. Do no try to access it.
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
  if (core_->type_ != Value::Category::Scalar)                                                                         \
    NTA_THROW << "value not found.";                                                                                   \
  errno = 0;                                                                                                           \
  char *end;                                                                                                           \
  T val = (T)I;                                                                                                        \
  if (errno)                                                                                                           \
    NTA_THROW << "In '" << core_->scalar_ << "' numeric conversion error: " << std::strerror(errno);                   \
  if (*end != '\0')                                                                                                    \
    NTA_THROW << "In '" << core_->scalar_ << "' numeric conversion error: invalid char.";                              \
  return val;

int8_t Value::asInt8() const { NTA_CONVERT(int8_t, std::strtol(core_->scalar_.c_str(), &end, 0)); }
int16_t Value::asInt16() const { NTA_CONVERT(int16_t, std::strtol(core_->scalar_.c_str(), &end, 0)); }
uint16_t Value::asUInt16() const { NTA_CONVERT(uint16_t, std::strtoul(core_->scalar_.c_str(), &end, 0)); }
int32_t Value::asInt32() const { NTA_CONVERT(int32_t, std::strtol(core_->scalar_.c_str(), &end, 0)); }
uint32_t Value::asUInt32() const { NTA_CONVERT(uint32_t, std::strtoul(core_->scalar_.c_str(), &end, 0)); }
int64_t Value::asInt64() const { NTA_CONVERT(int64_t, std::strtoll(core_->scalar_.c_str(), &end, 0)); }
uint64_t Value::asUInt64() const { NTA_CONVERT(uint64_t, std::strtoull(core_->scalar_.c_str(), &end, 0)); }
float Value::asFloat() const { NTA_CONVERT(float, std::strtof(core_->scalar_.c_str(), &end)); }
double Value::asDouble() const { NTA_CONVERT(double, std::strtod(core_->scalar_.c_str(), &end)); }
std::string Value::asString() const {
  if (core_->type_ != Value::Category::Scalar)
    NTA_THROW << "value not found.";
  return core_->scalar_;
}
bool Value::asBool() const {
  if (core_->type_ != Value::Category::Scalar)
    NTA_THROW << "value not found. " << core_->scalar_;
  std::string val = str();
  transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val == "true" || val == "on" || val == "1" || val == "yes")
    return true;
  if (val == "false" || val == "off" || val == "0" || val == "no")
    return false;
  NTA_THROW << "Invalid value for a boolean. " << val;
}

/**
 * a local function to apply escapes for a JSON string.
 See: http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf
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
      if (*c <= '\x1f' || *c == '\x7f') { 
        //control characters -> convert to hex.
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
    o << " |"; // all blanks are significant
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

static void to_yaml(std::ostream &f, const htm::Value &v, std::string indent) {
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

// Keep this around for debugging.  Useful for unit tests.
// A validation routine -- only useful validating that all internal links are valid.
// to call:   status = vm.check();
bool Value::check(const std::string& indent) const {
  bool status = true;
  if (core_->type_ == Scalar) {
    // std::cout << indent << "Scalar(value=\"" << core_->scalar_ << "\")\n";
  } else if (core_->type_ == Empty) {
    // std::cout << indent << "Empty(value=\"" << core_->scalar_ << "\)\n";
    if (core_->parent_) {
      std::cout << indent << "ERROR: No Zombies should be in the tree.\n";
      status = false;
    }
  } else if (core_->type_ == Sequence) {
    //std::cout << indent << "Sequence()\n";
  } else if (core_->type_ == Map) {
    //std::cout << indent << "Map()\n";
  }

  for (size_t i = 0; i < core_->vec_.size(); i++) {
    //std::cout << indent << "  - index(" << core_->vec_[i]->second.core_->index_ << ")  key(\"" << core_->vec_[i]->first << "\")\n";
    if (core_->vec_[i]->second.core_->index_ != i) {
      std::cout << indent << "i = " << i << "; ERROR: index_ does not match vec_ index\n";
      status = false;
    }
    if (core_->vec_[i]->second.core_->key_ != core_->vec_[i]->first) {
      std::cout << indent << "i = " << i << "; ERROR: key in node does not match map_ key\n";
      status = false;
    }
    if (core_->parent_.get() != core_->parent_.get()) {
      std::cout << indent << "ERROR: parents do not match.\n";
      status = false;
    }
    if (!core_->vec_[i]->second.check(indent + "  "))
      status = false;
  }
  return status;
}

namespace htm {
std::ostream &operator<<(std::ostream &f, const htm::Value &v) {
  f << v.to_json();
  return f;
}
} // namespace htm
