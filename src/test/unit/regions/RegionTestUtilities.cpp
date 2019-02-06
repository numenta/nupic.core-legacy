/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
 


 * Author: David Keeney, July, 2018
 * ---------------------------------------------------------------------
 */


#define VERBOSE  if (verbose)  std::cerr << "[          ] "


#include <iostream>

#include "RegionTestUtilities.hpp"

#include <nupic/engine/Output.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>

using namespace nupic;
using namespace nupic::algorithms::backtracking_tm;
namespace testing {


// a utility function to compare parameters in the spec with getters/setters.
// Assumes that the default value in the Spec is the same as the default when
// creating a region with default constructor.  The Set checks might not work if
// the range of a ReadWriteAccess parameter is restricted. In this case, add a constraint to Spec.
void checkGetSetAgainstSpec(Region_Ptr_t region1, size_t expectedSpecCount,bool verbose) {
  // check spec cache
  const std::shared_ptr<Spec> ns = region1->getSpec();
  ASSERT_TRUE(ns == region1->getSpec())
    << " First and second fetches of the spec gives different addresses. "
    << " It was not cached.";

  // Make sure the number of parameters matches.
  size_t specCount = ns->parameters.getCount();
  ASSERT_TRUE(specCount == expectedSpecCount)
      << "Unexpected number of parameters. Expected=" << expectedSpecCount
      << ", Actual=" << specCount;

  // Look through the Spec and look for every instance of a parameter.
  // get/set/verify each parameter
  bool negativeCheck = true;
  for (size_t i = 0; i < specCount; i++) {
    std::pair<string, ParameterSpec> p = ns->parameters.getByIndex(i);
    string name = p.first;
    try {
      if (p.second.count == 1) {
        switch (p.second.dataType) {

        case NTA_BasicType_UInt32: {
          if (negativeCheck) {
				    negativeCheck = false;
				    VERBOSE << "negative check..." << std::endl;
				    EXPECT_THROW(region1->getParameterInt32("numberOfCols"), nupic::Exception); 
			    }
          VERBOSE << "Parameter \"" << name << "\" type: " << BasicType::getName(p.second.dataType) << std::endl;

          // check the getter.
          UInt32 v = region1->getParameterUInt32(name);
          if (!p.second.defaultValue.empty()) {
            UInt32 d = std::stoul(p.second.defaultValue, nullptr, 0);
            EXPECT_TRUE(v == d)
                << "Failure: Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            if (p.second.constraints == "bool") {
              UInt32 d = (v) ? 0u : 1u;
              region1->setParameterUInt32(name, d); // change value
              UInt32 s = region1->getParameterUInt32(name);
              EXPECT_TRUE(s == d)
                  << "Parameter \"" << name
                  << "\" Actual value does not match changed value. Expected="
                  << (!v) << ", Actual=" << s;
            } else if (p.second.constraints == "") {
              region1->setParameterUInt32(name, 0xFFFFFFFF); // set max value
              UInt32 s = region1->getParameterUInt32(name);
              EXPECT_TRUE(s == 0xFFFFFFFF)
                  << "Failure: Parameter \"" << name
                  << "\" Actual value does not match UInt32 max. Expected="
                  << 0xFFFFFFFF << ", Actual=" << s;
            }
            region1->setParameterUInt32(name, v); // return to original value.
          }
          break;
        }

        case NTA_BasicType_Int32: {
          VERBOSE << "Parameter \"" << name << "\" type: " << BasicType::getName(p.second.dataType) << std::endl;
          // check the getter.
          Int32 v = region1->getParameterInt32(name);
          if (!p.second.defaultValue.empty()) {
            Int32 d = std::stoul(p.second.defaultValue, nullptr, 0);
            EXPECT_TRUE(v == d)
                << "Failure: Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            if (p.second.constraints == "") {
              region1->setParameterInt32(name, 0x7FFFFFFF); // set max value
              Int32 s = region1->getParameterInt32(name);
              EXPECT_TRUE(s == 0x7FFFFFFF)
                  << "Parameter \"" << name
                  << "\" Actual value does not match Int32 max. Expected="
                  << 0x7FFFFFFF << ", Actual=" << s;
              region1->setParameterInt32(name, v); // return to original value.
            }
          }
          break;
        }

        case NTA_BasicType_Real32: {
          VERBOSE << "Parameter \"" << name << "\" type: " << BasicType::getName(p.second.dataType) << std::endl;
          // check the getter.
          Real32 v = region1->getParameterReal32(name);
          if (!p.second.defaultValue.empty()) {
            Real32 d = std::strtof(p.second.defaultValue.c_str(), nullptr);
            EXPECT_TRUE(v == d)
                << "Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            if (p.second.constraints == "") {
              region1->setParameterReal32(name, FLT_MAX); // set max value
              Real32 s = region1->getParameterReal32(name);
              EXPECT_TRUE(s == FLT_MAX)
                  << "Parameter \"" << name
                  << "\" Actual value does not match float max. Expected="
                  << FLT_MAX << ", Actual=" << s;
              region1->setParameterReal32(name, v); // return to original value.
            }
          }
          break; 
        }

        case NTA_BasicType_Real64: {
          VERBOSE << "Parameter \"" << name << "\" type: " << BasicType::getName(p.second.dataType) << std::endl;
          // check the getter.
          Real64 v = region1->getParameterReal64(name);
          if (!p.second.defaultValue.empty()) {
            Real64 d = std::strtod(p.second.defaultValue.c_str(), nullptr);
            EXPECT_TRUE(v == d)
                << "Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            if (p.second.constraints == "") {
              region1->setParameterReal64(name, FLT_MAX); // set max value
              Real64 s = region1->getParameterReal64(name);
              EXPECT_TRUE(s == FLT_MAX)
                  << "Parameter \"" << name
                  << "\" Actual value does not match float max. Expected="
                  << FLT_MAX << ", Actual=" << s;
              region1->setParameterReal64(name, v); // return to original value.
            }
          }
          break;
        }

        case NTA_BasicType_Bool: {
          VERBOSE << "Parameter \"" << name << "\" type: " << BasicType::getName(p.second.dataType) << std::endl;
          // check the getter.
          bool v = region1->getParameterBool(name);
          if (!p.second.defaultValue.empty()) {
            bool d = (p.second.defaultValue == "true");
            EXPECT_TRUE(v == d)
                << "Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            region1->setParameterBool(name, !v); // change value
            bool s = region1->getParameterBool(name);
            EXPECT_TRUE(s == !v)
                << "Parameter \"" << name
                << "\" Actual value does not match changed value. Expected="
                << (!v) << ", Actual=" << s;
            region1->setParameterBool(name, v); // return to original value.
          }
          break;
        }

        default:
          FAIL() << "Parameter \"" << name << "\" Invalid data type.  found "
                  << p.second.dataType;
          break;
        } // end switch
      } else {

        // Array types
        switch (p.second.dataType) {
        case NTA_BasicType_Byte: {
          std::string v = region1->getParameterString(name);
          if (!p.second.defaultValue.empty()) {
            std::string d = p.second.defaultValue;
            EXPECT_STREQ(v.c_str(), d.c_str())
                << "Parameter \"" << name
                << "\" Actual value does not match default. Expected=" << d
                << ", Actual=" << v;
          }
          // check the setter.
          if (p.second.accessMode == ParameterSpec::ReadWriteAccess) {
            region1->setParameterString(name, "XXX"); // change value
            std::string s = region1->getParameterString(name);
            EXPECT_STREQ(s.c_str(), "XXX")
                << "Parameter \"" << name
                << "\" Actual value does not match changed value. Expected='XXX'"
                << ", Actual=" << s;
            region1->setParameterString(name, v); // return to original value.
          }
          break;
        }
        case NTA_BasicType_UInt32:  // array of UInt32
        case NTA_BasicType_Real32:  // array of Real32
          break;

        default:
          FAIL()
              << "Parameter \"" << name << "\" Invalid data type.  found "
              << BasicType::getName(p.second.dataType);
          break;
        } // end switch
      }
    } catch (nupic::Exception &ex) {
      FAIL()
             << "Exception while processing parameter " << name << ":  "
             << ex.getFilename() << "(" << ex.getLineNumber() << ") "
             << ex.getMessage();
    } catch (std::exception &e) {
      FAIL()
             << "Exception while processing parameter " << name << ":  "
             << e.what() << "" << std::endl;
    }
  } // end loop

}

// --- Tests the input/output access for a C++ implemented region against the Spec.
void checkInputOutputsAgainstSpec(Region_Ptr_t region1, bool verbose) 
{
  const std::shared_ptr<Spec> ns = region1->getSpec();

  for (Size i = 0; i < ns->outputs.getCount(); i++) {
    std::string name = ns->outputs.getByIndex(i).first;
    OutputSpec ospec = ns->outputs.getByIndex(i).second;
    
    VERBOSE << "Output \"" << name << "\" type: " << BasicType::getName(ospec.dataType) << std::endl;
    Output *O = region1->getOutput(name);
    ASSERT_TRUE(O != nullptr) << "The output obj could not be found.";
    EXPECT_TRUE(ospec.dataType == O->getData().getType())
      << "Output type for \"" << name << "\" does not match spec. "
      << "found " << BasicType::getName(O->getData().getType());
  } // end for outputs

  for (Size j = 0; j < ns->inputs.getCount(); j++) {
    std::string name = ns->inputs.getByIndex(j).first;
    InputSpec ispec = ns->inputs.getByIndex(j).second;
    
    VERBOSE << "Input \"" << name << "\" type: " << BasicType::getName(ispec.dataType) << std::endl;
    Input *I = region1->getInput(name);
    ASSERT_TRUE(I != nullptr) << "The input obj could not be found.";
    if (I->isInitialized()) {
      EXPECT_TRUE(ispec.dataType == I->getData().getType())
        << "Input type for \"" << name << "\" does not match spec. "
        << "found " << BasicType::getName(I->getData().getType());
    }
  } // end for inputs
}

// a utility function to compare two parameter arrays
::testing::AssertionResult compareParameterArrays(Region_Ptr_t region1,
                                                  Region_Ptr_t region2,
                                                  std::string parameter,
                                                  NTA_BasicType type) {
  UInt32 *buf1;
  UInt32 *buf2;
  Real32 *buf3;
  Real32 *buf4;
  Real64 *buf5;
  Real64 *buf6;
  Array array1(type);
  Array array2(type);
  region1->getParameterArray(parameter, array1);
  region2->getParameterArray(parameter, array2);


  if (type != array1.getType())
    return ::testing::AssertionFailure()
           << "Failure: Original Array1 for parameter '" << parameter
           << "' is not the expected type. expected: "
           << BasicType::getName(type) << ", found: "
          << BasicType::getName(array1.getType());
  if (type != array2.getType())
    return ::testing::AssertionFailure()
           << "Failure: Restored Array2 for parameter '" << parameter
           << "' is not the expected type. expected: "
           << BasicType::getName(type) << ", found: "
          << BasicType::getName(array1.getType());

  size_t len1 = array1.getCount();
  size_t len2 = array2.getCount();
  if (len1 != len2) {
    return ::testing::AssertionFailure()
           << "Failure: Arrays for parameter '" << parameter
           << "' are not the same length.";
  }
  switch (type) {
  case NTA_BasicType_UInt32:
    buf1 = (UInt32 *)array1.getBuffer();
    buf2 = (UInt32 *)array2.getBuffer();
    for (size_t i = 0; i < len1; i++) {
      if (buf1[i] != buf2[i]) {
        return ::testing::AssertionFailure()
               << "Failure: Array element for parameter '" << parameter << "["
               << i << "]' is not the same after restore.";
      }
    }
    break;

  case NTA_BasicType_Real32:
    buf3 = (Real32 *)array1.getBuffer();
    buf4 = (Real32 *)array2.getBuffer();
    for (size_t i = 0; i < len1; i++) {
      if (buf3[i] != buf4[i]) {
        return ::testing::AssertionFailure()
               << "Failure: Array element for parameter '" << parameter << "["
               << i << "]' is not the same after restore.";
      }
    }
    break;

  case NTA_BasicType_Real64:
    buf5 = (Real64 *)array1.getBuffer();
    buf6 = (Real64 *)array2.getBuffer();
    for (size_t i = 0; i < len1; i++) {
      if (buf5[i] != buf6[i]) {
        return ::testing::AssertionFailure()
               << "Failure: Array element for parameter '" << parameter << "["
               << i << "]' is not the same after restore.";
      }
    }
    break;
  default:
    break;
  } // end switch
  return ::testing::AssertionSuccess();
}

// uses the Spec to capture the non-array parameters and write them into the
// provided map.
::testing::AssertionResult
captureParameters(Region_Ptr_t region,
                  std::map<std::string, std::string> &parameters) {
  parameters.clear();
  const std::shared_ptr<Spec> ns = region->getSpec();
  size_t specCount = ns->parameters.getCount();
  // Look through the Spec and look for every instance of a parameter.
  // get each parameter, convert to a string and store in the parameters map.
  for (size_t i = 0; i < specCount; i++) {
    std::pair<std::string, ParameterSpec> p = ns->parameters.getByIndex(i);
    std::string name = p.first;

    try {
      if (p.second.count == 1) {
        switch (p.second.dataType) {
        case NTA_BasicType_Bool: {
          parameters[name] = (region->getParameterBool(name) ? "1" : "0");
          break;
        }
        case NTA_BasicType_UInt32: {
          parameters[name] = std::to_string(region->getParameterUInt32(name));
          break;
        }
        case NTA_BasicType_Int32: {
          parameters[name] = std::to_string(region->getParameterInt32(name));
          break;
        }
        case NTA_BasicType_Real32: {
          parameters[name] = std::to_string(region->getParameterReal32(name));
          break;
        }
        case NTA_BasicType_Byte: {
          parameters[name] = region->getParameterString(name);
          break;
        }
        default:
          break;
        } // end switch
      }
    } catch (nupic::Exception &ex) {
      return ::testing::AssertionFailure()
             << "nupic::Exception while processing parameter " << name << ":  "
             << ex.getFilename() << "(" << ex.getLineNumber() << ") "
             << ex.getMessage();
    } catch (std::exception &e) {
      return ::testing::AssertionFailure()
             << "Exception while processing parameter " << name << ":  "
             << e.what() << "" << std::endl;
    }
  } // end for
  return ::testing::AssertionSuccess();
}

// uses the Spec to find non-array parameters in the region and compare them to
// contents of the provided map.
::testing::AssertionResult
compareParameters(Region_Ptr_t region,
                  std::map<std::string, std::string> &parameters) {
  const std::shared_ptr<Spec> ns = region->getSpec();
  size_t specCount = ns->parameters.getCount();
  // Look through the Spec and look for every instance of a single parameter.
  // get each parameter, convert to a string and compare with what is stored in
  // the parameters map.
  for (size_t i = 0; i < specCount; i++) {
    std::pair<std::string, ParameterSpec> p = ns->parameters.getByIndex(i);
    std::string name = p.first;

    try {
      if (p.second.count == 1) {
        switch (p.second.dataType) {
        case NTA_BasicType_UInt32: {
          if (parameters[name] !=
              std::to_string(region->getParameterUInt32(name))) {
            return ::testing::AssertionFailure()
                   << "Parameter " << name << " Does not match.  Expected "
                   << parameters[name] << " found "
                   << std::to_string(region->getParameterUInt32(name)) << "";
          }
        } break;

        case NTA_BasicType_Int32: {
          if (parameters[name] !=
              std::to_string(region->getParameterInt32(name))) {
            return ::testing::AssertionFailure()
                   << "Parameter " << name << " Does not match.  Expected "
                   << parameters[name] << " found "
                   << std::to_string(region->getParameterInt32(name)) << "";
          }
        } break;

        case NTA_BasicType_Real32: {
          if (parameters[name] !=
              std::to_string(region->getParameterReal32(name))) {
            return ::testing::AssertionFailure()
                   << "Parameter " << name << " Does not match.  Expected "
                   << parameters[name] << " found "
                   << std::to_string(region->getParameterReal32(name)) << "";
          }
        } break;

        case NTA_BasicType_Bool: {
          if (parameters[name] !=
              (region->getParameterBool(name) ? "1" : "0")) {
            return ::testing::AssertionFailure()
                   << "Parameter " << name << " Does not match.  Expected "
                   << parameters[name] << " found "
                   << (region->getParameterBool(name) ? "1" : "0") << "";
          }
        } break;

        case NTA_BasicType_Byte: {
          if (parameters[name] != region->getParameterString(name)) {
            return ::testing::AssertionFailure()
                   << "Parameter " << name << " Does not match.  Expected "
                   << parameters[name] << " found "
                   << region->getParameterString(name) << "";
          }
        } break;

        default:
          break;
        } // end switch
      }
    } catch (nupic::Exception &ex) {
      return ::testing::AssertionFailure()
             << "nupic::Exception while processing parameter " << name << ":  "
             << ex.getFilename() << "(" << ex.getLineNumber() << ") "
             << ex.getMessage();
    } catch (std::exception &e) {
      return ::testing::AssertionFailure()
             << "Exception while processing parameter " << name << ":  "
             << e.what() << "" << std::endl;
    }
  } // end for
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult
compareOutputs(Region_Ptr_t region1, Region_Ptr_t region2, std::string name) {
  // Compare the Array objects.
  if (region1->getOutput(name)->getData() ==
      region2->getOutput(name)->getData())
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure();
}


} // namespace testing
