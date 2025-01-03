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

/** @file
Implementation of Spec API
*/

#include <nupic/engine/Spec.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

Spec::Spec() : singleNodeOnly(false), description("") {}
bool Spec::operator==(const Spec &o) const {
  if (singleNodeOnly != o.singleNodeOnly || description != o.description ||
      parameters != o.parameters || outputs != o.outputs ||
      inputs != o.inputs || commands != o.commands) {
    return false;
  }
  return true;
}
std::string Spec::getDefaultInputName() const {
  if (inputs.getCount() == 0)
    return "";
  if (inputs.getCount() == 1)
    return inputs.getByIndex(0).first;

  // search for default input, but detect multple defaults
  bool found = false;
  std::string name;

  for (size_t i = 0; i < inputs.getCount(); ++i) {
    const std::pair<std::string, InputSpec> &p = inputs.getByIndex(i);
    if (p.second.isDefaultInput) {
      NTA_CHECK(!found)
          << "Internal error -- multiply-defined default inputs in Spec";
      found = true;
      name = p.first;
    }
  }
  NTA_CHECK(found)
      << "Internal error -- multiple inputs in Spec but no default";
  return name;
}

std::string Spec::getDefaultOutputName() const {
  if (outputs.getCount() == 0)
    return "";
  if (outputs.getCount() == 1)
    return outputs.getByIndex(0).first;

  // search for default output, but detect multple defaults
  bool found = false;
  std::string name;

  for (size_t i = 0; i < outputs.getCount(); ++i) {
    const std::pair<std::string, OutputSpec> &p = outputs.getByIndex(i);
    if (p.second.isDefaultOutput) {
      NTA_CHECK(!found)
          << "Internal error -- multiply-defined default outputs in Spec";
      found = true;
      name = p.first;
    }
  }
  NTA_CHECK(found)
      << "Internal error -- multiple outputs in Spec but no default";
  return name;
}

InputSpec::InputSpec(std::string description, NTA_BasicType dataType,
                     UInt32 count, bool required, bool regionLevel,
                     bool isDefaultInput, bool requireSplitterMap, bool sparse)
    : description(std::move(description)), dataType(dataType), count(count),
      required(required), regionLevel(regionLevel),
      isDefaultInput(isDefaultInput), requireSplitterMap(requireSplitterMap),
      sparse(sparse) {}
bool InputSpec::operator==(const InputSpec &o) const {
  return required == o.required && regionLevel == o.regionLevel &&
         isDefaultInput == o.isDefaultInput && sparse == o.sparse &&
         requireSplitterMap == o.requireSplitterMap && dataType == o.dataType &&
         count == o.count && description == o.description;
}
OutputSpec::OutputSpec(std::string description, NTA_BasicType dataType,
                       size_t count, bool regionLevel, bool isDefaultOutput,
                       bool sparse)
    : description(std::move(description)), dataType(dataType), count(count),
      regionLevel(regionLevel), isDefaultOutput(isDefaultOutput),
      sparse(sparse) {}
bool OutputSpec::operator==(const OutputSpec &o) const {
  return regionLevel == o.regionLevel && isDefaultOutput == o.isDefaultOutput &&
         sparse == o.sparse && dataType == o.dataType && count == o.count &&
         description == o.description;
}

CommandSpec::CommandSpec(std::string description)
    : description(std::move(description)) {}
bool CommandSpec::operator==(const CommandSpec &o) const {
  return description == o.description;
}

ParameterSpec::ParameterSpec(std::string description, NTA_BasicType dataType,
                             size_t count, std::string constraints,
                             std::string defaultValue, AccessMode accessMode)
    : description(std::move(description)), dataType(dataType), count(count),
      constraints(std::move(constraints)),
      defaultValue(std::move(defaultValue)), accessMode(accessMode) {
  // Parameter of type byte is not supported;
  // Strings are specified as type byte, length = 0
  if (dataType == NTA_BasicType_Byte && count > 0)
    NTA_THROW << "Parameters of type 'byte' are not supported";
}
bool ParameterSpec::operator==(const ParameterSpec &o) const {
  return dataType == o.dataType && count == o.count &&
         description == o.description && constraints == o.constraints &&
         defaultValue == o.defaultValue && accessMode == o.accessMode;
}

std::string Spec::toString() const {
  // TODO -- minimal information here; fill out with the rest of
  // the parameter spec information
  std::stringstream ss;
  ss << "Spec:"
     << "\n";
  ss << "Description:"
     << "\n"
     << this->description << "\n"
     << "\n";

  ss << "Parameters:"
     << "\n";
  for (size_t i = 0; i < parameters.getCount(); ++i) {
    const std::pair<std::string, ParameterSpec> &item =
        parameters.getByIndex(i);
    ss << "  " << item.first << "\n"
       << "     description: " << item.second.description << "\n"
       << "     type: " << BasicType::getName(item.second.dataType) << "\n"
       << "     count: " << item.second.count << "\n";
  }

  ss << "Inputs:"
     << "\n";
  for (size_t i = 0; i < inputs.getCount(); ++i) {
    ss << "  " << inputs.getByIndex(i).first << "\n";
  }

  ss << "Outputs:"
     << "\n";
  for (size_t i = 0; i < outputs.getCount(); ++i) {
    ss << "  " << outputs.getByIndex(i).first << "\n";
  }

  ss << "Commands:"
     << "\n";
  for (size_t i = 0; i < commands.getCount(); ++i) {
    ss << "  " << commands.getByIndex(i).first << "\n";
  }

  return ss.str();
}

} // namespace nupic
