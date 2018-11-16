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

#include <nupic/engine/Spec.hpp>
#include <nupic/engine/YAMLUtils.hpp>
#include <nupic/ntypes/Collection.hpp>
#include <nupic/ntypes/MemStream.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/types/BasicType.hpp>
#include <string.h> // strlen
#include <yaml-cpp/yaml.h>

#include <sstream>

namespace nupic {
namespace YAMLUtils {

/*
 * These functions are used internally by toValue and toValueMap
 */
static void _toScalar(const YAML::Node& node, std::shared_ptr<Scalar>& s);
static void _toArray(const YAML::Node& node, std::shared_ptr<Array>& a);
static Value toValue(const YAML::Node& node, NTA_BasicType dataType);

static void _toScalar(const YAML::Node &node, std::shared_ptr<Scalar> &s) {
  NTA_CHECK(node.Type() == YAML::NodeType::Scalar);
  switch (s->getType()) {
  case NTA_BasicType_Byte:
    // We should have already detected this and gone down the string path
    NTA_THROW << "Internal error: attempting to convert YAML string to scalar of type Byte";
    break;
  case NTA_BasicType_UInt16:
    s->value.uint16 = node.as<UInt16>();
    break;
  case NTA_BasicType_Int16:
    s->value.int16 = node.as<Int16>();
    break;
  case NTA_BasicType_UInt32:
    s->value.uint32 = node.as<UInt32>();
    break;
  case NTA_BasicType_Int32:
    s->value.int32 = node.as<Int32>();
    break;
  case NTA_BasicType_UInt64:
    s->value.uint64 = node.as<UInt64>();
    break;
  case NTA_BasicType_Int64:
    s->value.int64 = node.as<Int64>();
    break;
  case NTA_BasicType_Real32:
    s->value.real32 = node.as<Real32>();
    break;
  case NTA_BasicType_Real64:
    s->value.real64 = node.as<Real64>();
    break;
  case NTA_BasicType_Bool:
    s->value.boolean = node.as<bool>();
    break;
  case NTA_BasicType_Handle:
    NTA_THROW << "Attempt to specify a YAML value for a scalar of type Handle";
    break;
  default:
    // should not happen
    const std::string val = node.as<std::string>();
    NTA_THROW << "Unknown data type " << s->getType() << " for yaml node '" << val << "'";
  }
}

static void _toArray(const YAML::Node& node, std::shared_ptr<Array>& a) {
  NTA_CHECK(node.Type() == YAML::NodeType::Sequence);

  a->allocateBuffer(node.size());
  void *buffer = a->getBuffer();

  for (size_t i = 0; i < node.size(); i++) {
    const YAML::Node &item = node[i];
    NTA_CHECK(item.Type() == YAML::NodeType::Scalar);
    switch (a->getType()) {
    case NTA_BasicType_Byte:
      // We should have already detected this and gone down the string path
      NTA_THROW << "Internal error: attempting to convert YAML string to array "
                   "of type Byte";
      break;
    case NTA_BasicType_UInt16:
      ((UInt16*)buffer)[i] = item.as<UInt16>();
      break;
    case NTA_BasicType_Int16:
      ((Int16*)buffer)[i] = item.as<Int16>();
      break;
    case NTA_BasicType_UInt32:
     ((UInt32*)buffer)[i] = item.as<UInt32>();
      break;
    case NTA_BasicType_Int32:
     ((Int32*)buffer)[i] = item.as<Int32>();
      break;
    case NTA_BasicType_UInt64:
     ((UInt64*)buffer)[i] = item.as<UInt64>();
      break;
    case NTA_BasicType_Int64:
     ((Int64*)buffer)[i] = item.as<Int64>();
      break;
    case NTA_BasicType_Real32:
     ((Real32*)buffer)[i] = item.as<Real32>();
      break;
    case NTA_BasicType_Real64:
     ((Real64*)buffer)[i] = item.as<Real64>();
      break;
    case NTA_BasicType_Bool:
     ((bool*)buffer)[i] = item.as<bool>();
      break;
    default:
      // should not happen
      NTA_THROW << "Unknown data type " << a->getType();
    }
  }
}

static Value toValue(const YAML::Node &node, NTA_BasicType dataType) {
  if (node.Type() == YAML::NodeType::Map ||
      node.Type() == YAML::NodeType::Null) {
    NTA_THROW << "YAML string does not represent a value.";
  }
  if (node.Type() == YAML::NodeType::Scalar) {
    if (dataType == NTA_BasicType_Byte) {
      // node >> *str;
      const std::string val = node.as<std::string>();
      Value v(val);
      return v;
    } else {
      std::shared_ptr<Scalar> s(new Scalar(dataType));
      _toScalar(node, s);
      Value v(s);
      return v;
    }
  } else {
    // array
    std::shared_ptr<Array> a(new Array(dataType));
    _toArray(node, a);
    Value v(a);
    return v;
  }
}

/*
 * For converting default values specified in nodespec string
 */
Value toValue(const std::string& yamlstring, NTA_BasicType dataType)
{
  // TODO -- return value? exceptions?
  const YAML::Node doc = YAML::Load(yamlstring);
  return toValue(doc, dataType);
}

/*
 * For converting param specs for Regions and LinkPolicies
 */
ValueMap toValueMap(const char *yamlstring,
                    Collection<ParameterSpec> &parameters,
                    const std::string &nodeType,
                    const std::string &regionName) {

  ValueMap vm;

  std::string paddedstring(yamlstring);
  // TODO: strip white space to determine if empty
  bool empty = (paddedstring.size() == 0);

  // TODO: utf-8 compatible?
  const YAML::Node doc = YAML::Load(paddedstring);
  if(!empty) {
    // A ValueMap is specified as a dictionary
    if (doc.Type() != YAML::NodeType::Map) {
      std::string ys(yamlstring);
      if (ys.size() > 30) {
        ys = ys.substr(0, 30) + "...";
      }
      NTA_THROW
          << "YAML string '" << ys
          << "' does not not specify a dictionary of key-value pairs. "
          << "Region and Link parameters must be specified as a dictionary";
    }
  }
  // Grab each value out of the YAML dictionary and put into the ValueMap
  // if it is allowed by the nodespec.
  for (auto i = doc.begin(); i != doc.end(); i++)
  {
    const auto key = i->first.as<std::string>();
    if (!parameters.contains(key))
    {
      std::stringstream ss;
      for (UInt j = 0; j < parameters.getCount(); j++){
        ss << "   " << parameters.getByIndex(j).first << "\n";
      }

      if (nodeType == std::string("")) {
        NTA_THROW << "Unknown parameter '" << key << "'\n"
                  << "Valid parameters are:\n" << ss.str();
      } else {
        NTA_CHECK(regionName != std::string(""));
        NTA_THROW << "Unknown parameter '" << key << "' for region '"
                  << regionName << "' of type '" << nodeType << "'\n"
                  << "Valid parameters are:\n"
                  << ss.str();
      }
    }
    if (vm.contains(key))
      NTA_THROW << "Parameter '" << key << "' specified more than once in YAML document";
    ParameterSpec ps = parameters.getByName(key); // makes a copy of ParameterSpec
    try
    {
      if (ps.accessMode == ParameterSpec::ReadOnlyAccess) {
        NTA_THROW << "Parameter '" << key << "'. This is ReadOnly access. Cannot be set.";
      }
      Value v = toValue(i->second, ps.dataType);
      if (v.isScalar() && ps.count != 1)
      {
        NTA_THROW << "Parameter '" << key << "'. Bad value in runtime parameters. Expected array value but got scalar value";
      }
      if (!v.isScalar() && ps.count == 1)
      {
        NTA_THROW << "Parameter '" << key << "'. Bad value in runtime parameters. Expected scalar value but got array value";
      }
      vm.add(key, v);
    } catch (std::runtime_error &e) {
      NTA_THROW << "Unable to set parameter '" << key << "'. " << e.what();
    }
  } //end for

  // Populate ValueMap with default values if they were not specified in the YAML dictionary.
  for (size_t i = 0; i < parameters.getCount(); i++)
  {
    const std::pair<std::string, ParameterSpec>& item = parameters.getByIndex(i);
    if (!vm.contains(item.first))
    {
      const ParameterSpec & ps = item.second;
      if (ps.defaultValue != "")
      {
        // TODO: This check should be uncommented after dropping NuPIC 1.x nodes (which don't comply) //FIXME try this
        // if (ps.accessMode != ParameterSpec::CreateAccess)
        // {
        //   NTA_THROW << "Default value for non-create parameter: " << item.first;
        // }

        try {
#ifdef YAMLDEBUG
          NTA_DEBUG << "Adding default value '" << ps.defaultValue
                    << "' to parameter " << item.first << " of type "
                    << BasicType::getName(ps.dataType) << " count " << ps.count;
#endif
          // NOTE: this can handle both scalers and arrays
          //       Arrays MUST be in Yaml sequence format even if one element.
          //       i.e.  [1,2,3]
          Value v = toValue(ps.defaultValue, ps.dataType);
          if (v.isScalar() && ps.count != 1)
          {
            NTA_THROW << "Parameter '" << item.first << "'. Bad default value in spec. Expected array value but got scalar value";
          }
          if (!v.isScalar() && ps.count == 1)
          {
            NTA_THROW << "Parameter '" << item.first << "'. Bad default value in spec. Expected scalar value but got array value";
          }
          vm.add(item.first, v);
        } catch (...) {
          NTA_THROW << "Unable to set default value for item '" << item.first
                    << "' of datatype " << BasicType::getName(ps.dataType)
                    << " with value '" << ps.defaultValue << "'";
        }
      }
    }
  }

  return vm;
}

} // end of YAMLUtils namespace
} // end of namespace nupic
