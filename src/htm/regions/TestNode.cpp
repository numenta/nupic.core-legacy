/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2017, Numenta, Inc.
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

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric> // std::accumulate
#include <sstream>


#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/regions/TestNode.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/utils/Log.hpp>
#include <htm/types/Types.hpp>


namespace htm {

TestNode::TestNode(const ValueMap &params, Region *region)
    : RegionImpl(region), computeCallback_(nullptr), nodeCount_(1)
{
  // params for get/setParameter testing
    // Populate the parameters with values.
  outputElementCount_ = params.getScalarT<UInt32>("count", 0);
  int32Param_ = params.getScalarT<Int32>("int32Param", 32);
  uint32Param_ = params.getScalarT<UInt32>("uint32Param", 33);
  int64Param_ = params.getScalarT<Int64>("int64Param", 64);
  uint64Param_ = params.getScalarT<UInt64>("uint64Param", 65);
  real32Param_ = params.getScalarT<Real32>("real32Param", 32.1f);
  real64Param_ = params.getScalarT<Real64>("real64Param", 64.1);
  boolParam_ = params.getScalarT<bool>("boolParam", false);


  shouldCloneParam_ = params.getScalarT<UInt32>("shouldCloneParam", 1) != 0;
  stringParam_ = params.getString("stringParam", "nodespec value");

  real32ArrayParam_.resize(8);
  for (size_t i = 0; i < 8; i++) {
    real32ArrayParam_[i] = float(i * 32);
  }

  int64ArrayParam_.resize(4);
  for (size_t i = 0; i < 4; i++) {
    int64ArrayParam_[i] = i * 64;
  }

  boolArrayParam_.resize(4);
  for (size_t i = 0; i < 4; i++) {
    boolArrayParam_[i] = (i % 2) == 1;
  }

  unclonedParam_.resize(nodeCount_);
  unclonedParam_[0] = params.getScalarT<UInt32>("unclonedParam", 0);

  possiblyUnclonedParam_.resize(nodeCount_);
  possiblyUnclonedParam_[0] =
      params.getScalarT<UInt32>("possiblyUnclonedParam", 0);

  unclonedInt64ArrayParam_.resize(nodeCount_);
  std::vector<Int64> v(4, 0); // length 4 vector, each element == 0
  unclonedInt64ArrayParam_[0] = v;

  // params used for computation
  delta_ = 1;
  iter_ = 0;
  ArWrapper arw;
}

TestNode::TestNode(ArWrapper& wrapper, Region *region)
  : RegionImpl(region), computeCallback_(nullptr), nodeCount_(1) 
{
  cereal_adapter_load(wrapper);
}


TestNode::~TestNode() {}


void TestNode::initialize() {
  bottomUpOut_ = getOutput("bottomUpOut");
  bottomUpIn_ = getInput("bottomUpIn");
  Dimensions dim = bottomUpOut_->getDimensions();
	NTA_CHECK(dim.isSpecified());
  // does not really handle dimensions > 2 right but this will do.
	nodeCount_ = 1;
  if (dim.size() > 1) {
  	nodeCount_ = dim.getCount()/dim[0];
  }
  outputElementCount_ = dim[0];

  unclonedParam_.resize(nodeCount_);
  for (unsigned int i = 1; i < nodeCount_; i++) {
    unclonedParam_[i] = unclonedParam_[0];
  }

  if (!shouldCloneParam_) {
    possiblyUnclonedParam_.resize(nodeCount_);
    for (unsigned int i = 1; i < nodeCount_; i++) {
      possiblyUnclonedParam_[i] = possiblyUnclonedParam_[0];
    }
  }

  unclonedInt64ArrayParam_.resize(nodeCount_);
  std::vector<Int64> v(4, 0); // length 4 vector, each element == 0
  for (unsigned int i = 1; i < nodeCount_; i++) {
    unclonedInt64ArrayParam_[i] = v;
  }
}

// This is the per-output buffer size
size_t TestNode::getNodeOutputElementCount(const std::string &outputName) const {
  if (outputName == "bottomUpOut") {
    return outputElementCount_;
  }
  return RegionImpl::getNodeOutputElementCount(outputName);  // default behavior
}

std::string TestNode::executeCommand(const std::vector<std::string> &args, Int64 index) {
  if (args.size() > 0) {
      for(auto n:args) {
        std::cout << "TestNode: args: " << n << std::endl;
      }
      std::string command = args[0];
      if (command == "HelloWorld") {
          NTA_CHECK(args.size() == 3) << "executeCommand(\"HelloWorld\") on TestNode requires "
                  "command plus 2 arguments, received command plus " << (args.size()-1);
          std::string result = "Hello World says: arg1="+args[1]+" arg2="+args[2];
          std::cout << "TestNode: result: " << result << std::endl;
          return result;
      }
  }
  return RegionImpl::executeCommand(args, index);
}



void TestNode::compute() {
  if (computeCallback_ != nullptr)
    computeCallback_(getName());

  bottomUpOut_ = getOutput("bottomUpOut");
  bottomUpIn_ = getInput("bottomUpIn");

  Array &outputArray = bottomUpOut_->getData();
  NTA_CHECK(outputArray.getCount() > 0) << "buffer not allocated.";
  NTA_CHECK(outputArray.getCount() == nodeCount_ * outputElementCount_)
       			<< "buffer size: " << outputArray.getCount()
				<< " expected: " << (nodeCount_ * outputElementCount_);
  NTA_CHECK(outputArray.getType() == NTA_BasicType_Real64);
  Real64 *baseOutputBuffer = (Real64 *)outputArray.getBuffer();

  // get the incoming buffer
  Array &inputArray = bottomUpIn_->getData();
  Real64* inputBuffer = (Real64*)inputArray.getBuffer();
  size_t count = inputArray.getCount();

    // trace facility
  NTA_DEBUG << "compute " << bottomUpIn_ << std::endl;

	
  // See TestNode.hpp for description of the computation
	
  Real64 *nodeOutputBuffer;
  for (UInt32 node = 0; node < nodeCount_; node++) {
    nodeOutputBuffer = baseOutputBuffer + node * outputElementCount_;

	  // output[0] = number of inputs + current iteration number
	  nodeOutputBuffer[0] = htm::Real64(count + iter_);

    if (outputArray.getCount() > 1) {
	    // output[n] = node + sum(inputs) + (n-1) * delta
      Real64 sum = 0.0;
      if (count > 0) {
        // simulate indexing by node
        size_t y = count / nodeCount_;
		    Real64 *start = inputBuffer + (node * y);
		    Real64 *end   = start + y;
	      sum = std::accumulate(start, end, 0.0);
      }
	    for (size_t i = 1; i < outputElementCount_; i++) {
	        nodeOutputBuffer[i] = node + sum + (i - 1) * delta_;
		  }
    }
  }

  // trace facility
  NTA_DEBUG << "compute " << bottomUpOut_ << "\n";


  iter_++;
}

Spec *TestNode::createSpec() {
  auto ns = new Spec;

  ns->description = "TestNode. Used as a plain simple plugin Region for unit tests only. "
      "This is not useful for any real applicaton.";

  /* ---- parameters ------ */
  ns->parameters.add( "count",
                     ParameterSpec(
							                   "Buffer size for bottomUpOut Output. "
                                 "Syntax: {count: 64}",  // description
	                               NTA_BasicType_UInt32,
							                   1,                         // elementCount 
							                   "",                        // constraints
							                   "",                        // defaultValue
							                   ParameterSpec::ReadWriteAccess));


  ns->parameters.add("int32Param",
                     ParameterSpec("Int32 scalar parameter", // description
                                   NTA_BasicType_Int32,
                                   1,    // elementCount
                                   "",   // constraints
                                   "32", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("uint32Param",
                     ParameterSpec("UInt32 scalar parameter", // description
                                   NTA_BasicType_UInt32,
                                   1,    // elementCount
                                   "",   // constraints
                                   "33", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("int64Param",
                     ParameterSpec("Int64 scalar parameter", // description
                                   NTA_BasicType_Int64,
                                   1,    // elementCount
                                   "",   // constraints
                                   "64", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("uint64Param",
                     ParameterSpec("UInt64 scalar parameter", // description
                                   NTA_BasicType_UInt64,
                                   1,    // elementCount
                                   "",   // constraints
                                   "65", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("real32Param",
                     ParameterSpec("Real32 scalar parameter", // description
                                   NTA_BasicType_Real32,
                                   1,      // elementCount
                                   "",     // constraints
                                   "32.1", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("real64Param",
                     ParameterSpec("Real64 scalar parameter", // description
                                   NTA_BasicType_Real64,
                                   1,      // elementCount
                                   "",     // constraints
                                   "64.1", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("boolParam",
                     ParameterSpec("bool scalar parameter", // description
                                   NTA_BasicType_Bool,
                                   1,       // elementCount
                                   "",      // constraints
                                   "false", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("real32ArrayParam",
                     ParameterSpec("int32 array parameter",
                                   NTA_BasicType_Real32,
                                   0, // array
                                   "", "", ParameterSpec::ReadWriteAccess));

  ns->parameters.add("int64ArrayParam",
                     ParameterSpec("int64 array parameter",  // description
					               NTA_BasicType_Int64,
                                   0, // array
                                   "", // constraints
								   "", // default Value
								   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("boolArrayParam",
                     ParameterSpec("bool array parameter", // description
					               NTA_BasicType_Bool,
                                   0, // array
                                   "", // constraints
								   "", // default Value
								   ParameterSpec::ReadOnlyAccess));

  ns->parameters.add("computeCallback",
                     ParameterSpec("address of a function that is called at every compute()",
                                   NTA_BasicType_UInt64,
					               1,  // element count
					               "", // constraints
                                   "", // handles must not have a default value
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("stringParam",
                     ParameterSpec("string parameter",
					               NTA_BasicType_Byte,
                                   0, // length=0 required for strings
                                   "",
								   "nodespec value",
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("unclonedParam",
                     ParameterSpec("has a separate value for each node", // description
                                   NTA_BasicType_UInt32,
                                   1,  // elementCount
                                   "", // constraints
                                   "", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("shouldCloneParam",
				      ParameterSpec("whether possiblyUnclonedParam should clone", // description
				                    NTA_BasicType_UInt32,
				                    1,            // elementCount
				                    "enum: 0, 1", // constraints
				                    "1",          // defaultValue
				                    ParameterSpec::ReadWriteAccess));

  ns->parameters.add("possiblyUnclonedParam",
				      ParameterSpec("cloned if shouldCloneParam is true", // description
				                    NTA_BasicType_UInt32,
				                    1,  // elementCount
				                    "", // constraints
				                    "", // defaultValue
				                    ParameterSpec::ReadWriteAccess));

  ns->parameters.add("unclonedInt64ArrayParam",
				      ParameterSpec("has a separate array for each node", // description
				                    NTA_BasicType_Int64,
				                    0,  // array                            //elementCount
				                    "", // constraints
				                    "", // defaultValue
				                    ParameterSpec::ReadWriteAccess));

  /* ----- inputs ------- */
  ns->inputs.add("bottomUpIn",
                 InputSpec("Primary input for the node",
				           NTA_BasicType_Real64,
                           0,     // count. omit?
                           true,  // required?
                           true, // isRegionLevel,
                           true   // isDefaultInput
                           ));

  /* ----- outputs ------ */
  ns->outputs.add("bottomUpOut",
                  OutputSpec("Primary output for the node",
                            NTA_BasicType_Real64,
                            0,     // count is dynamic
                            true, // isRegionLevel
                            true   // isDefaultOutput
                            ));

  /* ----- commands ------ */
  // commands TBD

  return ns;
}
Int32 TestNode::getParameterInt32(const std::string &name, Int64 index) {
  if (name == "count") {
    return outputElementCount_;
  }
  else if (name == "int32Param") {
  	return int32Param_;
  } else {
    return RegionImpl::getParameterInt32(name, index);
  }
}

UInt32 TestNode::getParameterUInt32(const std::string &name, Int64 index) {
  if (name == "uint32Param") {
    return uint32Param_;
  } else if (name == "unclonedParam") {
    if (index < 0) {
      NTA_THROW << "uncloned parameters cannot be accessed at region level";
    }
    return unclonedParam_[(size_t)index];
  } else if (name == "shouldCloneParam") {
    return ((UInt32)(shouldCloneParam_ ? 1 : 0));
  } else if (name == "possiblyUnclonedParam") {
    if (shouldCloneParam_) {
      return (possiblyUnclonedParam_[0]);
    } else {
      if (index < 0) {
        NTA_THROW << "uncloned parameters cannot be accessed at region level";
      }
      return (possiblyUnclonedParam_[(UInt)index]);
    }
  } else {
    return RegionImpl::getParameterUInt32(name, index);
  }
}
Int64 TestNode::getParameterInt64(const std::string &name, Int64 index) {
  if (name == "int64Param") {
    return int64Param_;
  } else {
    return RegionImpl::getParameterInt64(name, index);
  }
}
UInt64 TestNode::getParameterUInt64(const std::string &name, Int64 index) {
  if (name == "uint64Param") {
    return uint64Param_;
  } else if (name == "computeCallback") {
    return (UInt64)computeCallback_;
  } else {
    return RegionImpl::getParameterUInt64(name, index);
  }
}

Real32 TestNode::getParameterReal32(const std::string &name, Int64 index) {
  if (name == "real32Param") {
    return real32Param_;
  } else {
    return RegionImpl::getParameterReal32(name, index);
  }
}

Real64 TestNode::getParameterReal64(const std::string &name, Int64 index) {
  if (name == "real64Param") {
    return real64Param_;
  } else {
    return RegionImpl::getParameterReal64(name, index);
  }
}

bool TestNode::getParameterBool(const std::string &name, Int64 index) {
  if (name == "boolParam") {
    return boolParam_;
  } else {
    return RegionImpl::getParameterBool(name, index);
  }
}


std::string TestNode::getParameterString(const std::string &name, Int64 index) {
  if (name == "stringParam") {
    return stringParam_;
  } else {
    return RegionImpl::getParameterString(name, index);
  }
}

void TestNode::getParameterArray(const std::string &name, Int64 index, Array &array) {
  if (name == "int64ArrayParam") {
  	Array a(NTA_BasicType_Int64, &int64ArrayParam_[0], int64ArrayParam_.size());
	  array = a;
  }
  else if (name == "real32ArrayParam") {
  	Array a(NTA_BasicType_Real32, &real32ArrayParam_[0], real32ArrayParam_.size());
    array = a;
  } else if (name == "unclonedInt64ArrayParam") {
    if (index < 0) {
      NTA_THROW << "uncloned parameters cannot be accessed at region level";
    }
	  if (index >= (Int64)unclonedInt64ArrayParam_.size()) {
      NTA_THROW << "uncloned parameter index out of range";
    }
	  Array a(NTA_BasicType_Int64, &unclonedInt64ArrayParam_[static_cast<size_t>(index)][0], 
                                  unclonedInt64ArrayParam_[static_cast<size_t>(index)].size());
    array = a;
  } else {
    NTA_THROW << "TestNode::getParameterArray -- unknown parameter " << name;
  }
}



void TestNode::setParameterInt32(const std::string &name, Int64 index, Int32 value) {
  if (name == "count") {
     outputElementCount_ = value;
  }
  else if (name == "int32Param") {
    int32Param_ = value;
  } else {
	RegionImpl::setParameterInt32(name, index, value);
  }
}
void TestNode::setParameterUInt32(const std::string &name, Int64 index, UInt32 value) {
  if (name == "uint32Param") {
    uint32Param_ = value;
  }
  else if (name == "unclonedParam") {
    if (index < 0 || index >= static_cast<Int64>(unclonedParam_.size())) {
      NTA_THROW << "uncloned parameters index out of range";
    }
	unclonedParam_[static_cast<size_t>(index)] = value;
  }
  else if (name == "shouldCloneParam") {
      shouldCloneParam_ = !(value == 0);
  }
  else if (name == "possiblyUnclonedParam") {
    if (shouldCloneParam_) {
      possiblyUnclonedParam_[0] = value;
    } else {
      if (index < 0 || index >= static_cast<Int64>(possiblyUnclonedParam_.size())) {
        NTA_THROW << "uncloned parameters index out of range.";
      }
      possiblyUnclonedParam_[(size_t)index] = value;
    }
  } else {
	RegionImpl::setParameterUInt32(name, index, value);
  }
}

void TestNode::setParameterInt64(const std::string &name, Int64 index, Int64 value) {
  if (name == "int64Param") {
    int64Param_ = value;
  } else {
	RegionImpl::setParameterInt64(name, index, value);
  }
}

void TestNode::setParameterUInt64(const std::string &name, Int64 index, UInt64 value) {
  if (name == "uint64Param") {
    uint64Param_ = value;
  } else if (name == "computeCallback") {
    computeCallback_ = (computeCallbackFunc)value;
  } else {
	RegionImpl::setParameterUInt64(name, index, value);
  }
}

void TestNode::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  if (name == "real32Param") {
    real32Param_ = value;
  } else {
	RegionImpl::setParameterReal32(name, index, value);
  }
}

void TestNode::setParameterReal64(const std::string &name, Int64 index, Real64 value) {
  if (name == "real64Param") {
    real64Param_ = value;
  } else {
	RegionImpl::setParameterReal64(name, index, value);
  }
}

void TestNode::setParameterBool(const std::string &name, Int64 index, bool value) {
  if (name == "boolParam") {
    boolParam_ = value;
  } else {
	RegionImpl::setParameterBool(name, index, value);
  }
}


void TestNode::setParameterString(const std::string &name, Int64 index, const std::string& value) {
  if (name == "stringParam") {
    stringParam_ = value;
  } else {
	RegionImpl::setParameterString(name, index, value);
  }
}

void TestNode::setParameterArray(const std::string &name, Int64 index, const Array &array) {
  if (name == "int64ArrayParam" && array.getType() == NTA_BasicType_Int64) {
    int64ArrayParam_ = array.asVector<Int64>();
  }
  else if (name == "real32ArrayParam" && array.getType() == NTA_BasicType_Real32) {
    real32ArrayParam_ = array.asVector<Real32>();
  }
  else if (name == "unclonedParam" && array.getType() == NTA_BasicType_UInt32) {
    unclonedParam_ = array.asVector<UInt32>();
    if (index < 0) {
      NTA_THROW << "uncloned parameters cannot be accessed at region level";
    }
  }
  else if (name == "unclonedInt64ArrayParam" && array.getType() == NTA_BasicType_Int64) {
    if (index < 0) {
      NTA_THROW << "uncloned parameters cannot be accessed at region level";
    }
    unclonedInt64ArrayParam_[static_cast<Int64>(index)] = array.asVector<Int64>();
  }
  else {
    RegionImpl::setParameterArray(name, index, array);
  }
}


size_t TestNode::getParameterArrayCount(const std::string &name, Int64 index) {
  if (name == "int64ArrayParam") {
    return int64ArrayParam_.size();
  } else if (name == "real32ArrayParam") {
    return real32ArrayParam_.size();
  } else if (name == "boolArrayParam") {
    return boolArrayParam_.size();
  } else if (name == "unclonedInt64ArrayParam") {
    if (index < 0) {
      NTA_THROW << "uncloned parameters cannot be accessed at region level";
    }
    return unclonedInt64ArrayParam_[index].size();
  } else {
    NTA_THROW << "TestNode::getParameterArrayCount -- unknown parameter "
              << name;
  }
}

template <typename T>
static void arrayOut(std::ostream &s, const std::vector<T> &array,
                     const std::string &name) {
  s << "ARRAY_" << name << " ";
  s << array.size() << " ";
  for (auto elem : array) {
    s << elem << " ";
  }
}

template <typename T>
static void arrayIn(std::istream &s, std::vector<T> &array,
                    const std::string &name) {
  std::string expectedCookie = std::string("ARRAY_") + name;
  std::string cookie;
  s >> cookie;
  if (cookie != expectedCookie)
    NTA_THROW << "Bad cookie '" << cookie
              << "' for serialized array. Expected '" << expectedCookie << "'";
  size_t sz;
  s >> sz;
  array.resize(sz);
  for (size_t ix = 0; ix < sz; ix++) {
    s >> array[ix];
  }
}


  bool TestNode::operator==(const RegionImpl &o) const {
    if (o.getType() != "TestNode") return false;
    TestNode& other = (TestNode&)o;
    if (nodeCount_ != other.nodeCount_) return false;
    if (int32Param_ != other.int32Param_) return false;
    if (uint32Param_ != other.uint32Param_) return false;
    if (int64Param_ != other.int64Param_) return false;
    if (uint64Param_ != other.uint64Param_) return false;
    if (real32Param_ != other.real32Param_) return false;
    if (real64Param_ != other.real64Param_) return false;
    if (boolParam_ != other.boolParam_) return false;
    if (stringParam_ != other.stringParam_) return false;
    if (outputElementCount_ != other.outputElementCount_) return false;
    if (delta_ != other.delta_) return false;
    if (iter_ != other.iter_) return false;
    if (dim_ != other.dim_) return false;

    if (unclonedParam_.size() != other.unclonedParam_.size()) return false;
    for (size_t i = 0; i < unclonedParam_.size(); i++) {
      if (unclonedParam_[i] != other.unclonedParam_[i]) return false;
    }

    if (real32ArrayParam_.size() != other.real32ArrayParam_.size()) return false;
    for (size_t i = 0; i < real32ArrayParam_.size(); i++) {
      if (real32ArrayParam_[i] != other.real32ArrayParam_[i]) return false;
    }

    if (int64ArrayParam_.size() != other.int64ArrayParam_.size()) return false;
    for (size_t i = 0; i < int64ArrayParam_.size(); i++) {
      if (int64ArrayParam_[i] != other.int64ArrayParam_[i]) return false;
    }


    if (boolArrayParam_.size() != other.boolArrayParam_.size()) return false;
    for (size_t i = 0; i < boolArrayParam_.size(); i++) {
      if (boolArrayParam_[i] != other.boolArrayParam_[i]) return false;
    }

    if (shouldCloneParam_ != other.shouldCloneParam_) return false;
    if (possiblyUnclonedParam_.size() != other.possiblyUnclonedParam_.size()) return false;
    for (size_t i = 0; i < possiblyUnclonedParam_.size(); i++) {
      if (possiblyUnclonedParam_[i] != other.possiblyUnclonedParam_[i]) return false;
    }

    if (unclonedInt64ArrayParam_.size() != other.unclonedInt64ArrayParam_.size()) return false;
    for (size_t i = 0; i < unclonedInt64ArrayParam_.size(); i++)
    {
      if (unclonedInt64ArrayParam_[i].size() != other.unclonedInt64ArrayParam_[i].size())
        return false;
      for (size_t j = 0; j < unclonedInt64ArrayParam_[i].size(); j++) {
        if (unclonedInt64ArrayParam_[i][j] != other.unclonedInt64ArrayParam_[i][j])
          return false;
      }
    }

    return true;
  }



} // namespace htm
