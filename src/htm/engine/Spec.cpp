/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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

/** @file
Implementation of Spec API
*/

#include <htm/engine/Spec.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/utils/Log.hpp>
#include <htm/ntypes/Value.hpp>

namespace htm {

  
static bool startsWith_(const std::string &s, const std::string &prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}


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

InputSpec::InputSpec(std::string description,
                     NTA_BasicType dataType,
                     UInt32 count,
                     bool required,
                     bool regionLevel,
                     bool isDefaultInput)
    : description(std::move(description)), dataType(dataType), count(count),
      required(required), regionLevel(regionLevel),
      isDefaultInput(isDefaultInput) {}

bool InputSpec::operator==(const InputSpec &o) const {
  return required == o.required && regionLevel == o.regionLevel &&
         isDefaultInput == o.isDefaultInput &&
         dataType == o.dataType &&
         count == o.count && description == o.description;
}
std::ostream& operator<< (std::ostream& out, const InputSpec& self) {
   out << "      description: " << self.description << "\n"
       << "      type: " << BasicType::getName(self.dataType) << "\n"
       << "      count: " << self.count << "\n"
       << "      required: " << self.required << "\n"
       << "      regionLevel: " << self.regionLevel << "\n"
       << "      isDefaultInput: " << self.isDefaultInput << "\n";
   return out;
}

OutputSpec::OutputSpec(std::string description,
                       NTA_BasicType dataType,
                       size_t count,
                       bool regionLevel,
                       bool isDefaultOutput)
    : description(std::move(description)), dataType(dataType), count(count),
      regionLevel(regionLevel), isDefaultOutput(isDefaultOutput) {}

bool OutputSpec::operator==(const OutputSpec &o) const {
  return regionLevel == o.regionLevel && isDefaultOutput == o.isDefaultOutput &&
         dataType == o.dataType && count == o.count &&
         description == o.description;
}
std::ostream& operator<< (std::ostream& out, const OutputSpec& self) {
   out << "      description: " << self.description << "\n"
       << "      type: " << BasicType::getName(self.dataType) << "\n"
       << "      count: " << self.count << "\n"
       << "      regionLevel: " << self.regionLevel << "\n"
       << "      isDefaultInput: " << self.isDefaultOutput << "\n";
   return out;
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
}
bool ParameterSpec::operator==(const ParameterSpec &o) const {
  return dataType == o.dataType && count == o.count &&
         description == o.description && constraints == o.constraints &&
         defaultValue == o.defaultValue && accessMode == o.accessMode;
}
std::ostream& operator<< (std::ostream& out, const ParameterSpec& self) {
    out << "      description: " << self.description << "\n"
        << "      type: " << BasicType::getName(self.dataType) << "\n"
        << "      count: " << self.count << "\n"
        << "      access: ";
    switch(self.accessMode) {
    case ParameterSpec::CreateAccess:  out << "CreateAccess\n"; break;
    case ParameterSpec::ReadOnlyAccess: out << "ReadOnlyAccess\n"; break;
    case ParameterSpec::ReadWriteAccess: out << "ReadWriteAccess\n"; break;
    default: out << "UnknownAccess\n"; break;
    }
    out << "      defaultValue: " << self.defaultValue << "\n";
    out << "      constraints: " << self.constraints << "\n";
    return out;
}

std::string Spec::toString() const {
  // TODO -- minimal information here; fill out with the rest of
  // the parameter spec information
  std::stringstream ss;
  ss << "Spec: " << this->name  << "\n";
  ss << "  Description:"  << this->description << "\n";

  ss << "  Parameters:" << "\n";
  for (size_t i = 0; i < parameters.getCount(); ++i) {
    const std::pair<std::string, ParameterSpec> &item = parameters.getByIndex(i);
    ss << "    " << item.first << ":\n";
    ss << item.second << "\n";
  }

  ss << "  Inputs:" << "\n";
  for (size_t i = 0; i < inputs.getCount(); ++i) {
    const std::pair<std::string, InputSpec> &item = inputs.getByIndex(i);
    ss << "    " << item.first << ":\n";
    ss << item.second << "\n";
  }

  ss << "Outputs:" << "\n";
  for (size_t i = 0; i < outputs.getCount(); ++i) {
    const std::pair<std::string, OutputSpec> &item = outputs.getByIndex(i);
    ss << "    " << item.first << ":\n";
    ss << item.second << "\n";
  }

  if (commands.getCount() > 0) {
    ss << "  Commands:" << "\n";
    for (size_t i = 0; i < commands.getCount(); ++i) {
      ss << "    " << commands.getByIndex(i).first << ": "
         << commands.getByIndex(i).second.description << "\n";
    }
  }
  return ss.str();
}

std::ostream& operator<< (std::ostream& out, const Spec& self) {
  out << self.toString() << std::endl;
  return out;
}



// This is allows an alternate format in which to provide the Spec data.
// This will parse a Yaml or JSON string and populate the Spec structure.
// Field order does not matter.
//
// Expected yaml layout:
//    name: <the region's name>                                     (string, default "")
//    description: "text describing the region."                    (string, default "")
//    parameters:
//      <name>:                                         repeat for each parameter
//        description: "description of field",                      (string, default "")
//        type:  xxx,  <see BasicType::parse()>                     (enum, required)
//        count: <"0" if variable array, "1" if scalar>             (size_t, default "1")
//        constraints: <constraint text>,                           (string, default "" )
//        defaultValue: <as quoted JSON string>,                    (string, default "0")
//        access: <one of "Create", "ReadOnly", "ReadWrite">        (enum, default "Create")
//    inputs:
//      <name>:                                         repeat for each input
//        description: "description of input",                      (string, default "")
//        type:  xxx,   <see BasicType::parse()>                    (enum, required)
//        count: <"0" if variable array width, else fixed width>    (size_t, default "0")
//        required: <true if field must be provided>,               (bool, default "false")
//        isRegionLevel: <if true, propogates dimensions to region>,(bool, default "false")
//        isDefaultInput: <if true, assumed input for region>       (bool, default "false")
//    outputs:
//      <name>:                                         repeat for each output
//        description: "description of output",                     (string, default "")
//        type:  xxx,   <see BasicType::parse()>                    (enum, required)
//        count: <"0" if variable array width, else fixed width>    (size_t, default "0")
//        isRegionLevel: <true if inherits region dimensions>,      (bool, default "false")
//        isDefaultOutput: <if true, assumed output for region>     (bool, default "false")
//    commands:
//      <name>:                                         repeat for each command
//        description: "description of command",                    (string, default "")
//
void Spec::parseSpec(const std::string &yaml) {
  Value tree;
  tree.parse(yaml);

  NTA_CHECK(tree.isMap()) << "Invalid format for spec.";
  for (auto category_pair : tree) {
    std::string category = category_pair.first;
    std::string the_name;
    std::string itm;
    try {
      if (category == "name") {
        name = category_pair.second.str();
      } else if (category == "description")
        description = category_pair.second.str();
      else if (category == "singleNodeOnly")
        singleNodeOnly = category_pair.second.as<bool>();
      else if (category == "parameters") {
        for (auto parameter_pair : category_pair.second) {
          the_name = parameter_pair.first;
          ParameterSpec par;
          for (auto field_pair : parameter_pair.second) {
            itm = field_pair.first;
            if (itm == "description")
              par.description = field_pair.second.str();
            else if (itm == "type")
              par.dataType = BasicType::parse(field_pair.second.str());
            else if (itm == "count")
              par.count = field_pair.second.as<size_t>();
            else if (itm == "constraints")
              par.constraints = field_pair.second.str();
            else if (itm == "default")
              par.defaultValue = field_pair.second.str();
            else if (itm == "access") {
              std::string access = field_pair.second.str();
              if (startsWith_(access, "Create"))
                par.accessMode = ParameterSpec::CreateAccess;
              else if (startsWith_(access, "ReadOnly"))
                par.accessMode = ParameterSpec::ReadOnlyAccess;
              else if (startsWith_(access, "ReadWrite"))
                par.accessMode = ParameterSpec::ReadWriteAccess;
              else
                NTA_THROW << "Unexpected Access Mode: '" << access << "' in spec for " << name << "::";
            } else
              NTA_THROW << "Unexpected field item: '" << itm << "' in spec for " << name << "::";
          }
          parameters.add(the_name, par);
        }
      } else if (category == "inputs") {
        for (auto input_pair : category_pair.second) {
          the_name = input_pair.first;
          InputSpec par;
          for (auto field_pair : input_pair.second) {
            std::string itm = field_pair.first;
            if (itm == "description")
              par.description = field_pair.second.str();
            else if (itm == "type")
              par.dataType = BasicType::parse(field_pair.second.str());
            else if (itm == "count")
              par.count = field_pair.second.as<UInt32>();
            else if (itm == "required")
              par.required = field_pair.second.as<bool>();
            else if (itm == "isRegionLevel" || itm == "regionLevel")
              par.regionLevel = field_pair.second.as<bool>();
            else if (itm == "isDefaultInput")
              par.isDefaultInput = field_pair.second.as<bool>();
            else
              NTA_THROW << "Unexpected item: '" << itm << "' in spec for " << name << "::"
                        << "inputs '" << the_name << "'.";
          }
          inputs.add(the_name, par);
        }
      } else if (category == "outputs") {
        for (auto output_pair : category_pair.second) {
          the_name = output_pair.first;
          OutputSpec par;
          for (auto field_pair : output_pair.second) {
            itm = field_pair.first;
            if (itm == "description")
              par.description = field_pair.second.str();
            else if (itm == "type")
              par.dataType = BasicType::parse(field_pair.second.str());
            else if (itm == "count")
              par.count = field_pair.second.as<size_t>();
            else if (itm == "isRegionLevel" || itm == "regionLevel")
              par.regionLevel = field_pair.second.as<bool>();
            else if (itm == "isDefaultOutput")
              par.isDefaultOutput = field_pair.second.as<bool>();
            else
              NTA_THROW << "Unexpected item: '" << itm << "' in spec for " << name << "::"
                        << "outputs '" << the_name << "'.";
          }
          outputs.add(the_name, par);
        }
      } else
        NTA_THROW << "Unexpected category: in spec for " << name << "::.";
    } catch (const Exception &e) {
      std::string err = "Exception: '" + category + "', '" + the_name + "', '" + itm + "', " + std::string(e.what());
      std::cerr << err << std::endl;
      NTA_THROW << err;
    }
  }
}



} // namespace htm
