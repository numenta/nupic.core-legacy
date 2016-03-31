/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of the ScalarSensor
 */

#include <string>

// Workaround windows.h collision:
// https://github.com/sandstorm-io/capnproto/issues/213
#undef VOID
#include <capnp/any.h>

#include <nupic/encoders/ScalarSensor.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/ntypes/ObjectModel.hpp> // IWrite/ReadBuffer
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Input.hpp>


using capnp::AnyPointer;

namespace nupic
{
  ScalarSensor::ScalarSensor(const ValueMap& params, Region *region)
    : RegionImpl(region)
  {
    const UInt32 n = params.getScalarT<UInt32>("n");
    const UInt32 w = params.getScalarT<UInt32>("w");
    const Real64 resolution = params.getScalarT<Real64>("resolution");
    const Real64 radius = params.getScalarT<Real64>("radius");
    const Real64 minValue = params.getScalarT<Real64>("minValue");
    const Real64 maxValue = params.getScalarT<Real64>("maxValue");
    const bool periodic = params.getScalarT<bool>("periodic");
    const bool clipInput = params.getScalarT<bool>("clipInput");
    if (periodic)
    {
      encoder_ = new PeriodicScalarEncoder(w, minValue, maxValue, n, radius,
                                           resolution);
    }
    else
    {
      encoder_ = new ScalarEncoder(w, minValue, maxValue, n, radius, resolution,
                                   clipInput);
    }

    sensedValue_ = params.getScalarT<Real64>("sensedValue");
  }

  ScalarSensor::ScalarSensor(BundleIO& bundle, Region* region) :
    RegionImpl(region)
  {
    deserialize(bundle);
  }


  ScalarSensor::ScalarSensor(AnyPointer::Reader& proto, Region* region) :
    RegionImpl(region)
  {
    read(proto);
  }


  ScalarSensor::~ScalarSensor()
  {
    delete encoder_;
  }

  void ScalarSensor::compute()
  {
    Real32* array = (Real32*)encodedOutput_->getData().getBuffer();
    const Int32 iBucket = encoder_->encodeIntoArray(sensedValue_, array);
    ((Int32*)bucketOutput_->getData().getBuffer())[0] = iBucket;
  }

  /* static */ Spec*
  ScalarSensor::createSpec()
  {
    auto ns = new Spec;

    ns->singleNodeOnly = true;

    /* ----- parameters ----- */
    ns->parameters.add(
      "sensedValue",
      ParameterSpec(
        "Scalar input",
        NTA_BasicType_Real64,
        1, // elementCount
        "", // constraints
        "-1", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "n",
      ParameterSpec(
        "The length of the encoding",
        NTA_BasicType_UInt32,
        1, // elementCount
        "", // constraints
        "0", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "w",
      ParameterSpec(
        "The number of active bits in the encoding",
        NTA_BasicType_UInt32,
        1, // elementCount
        "", // constraints
        "0", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "resolution",
      ParameterSpec(
        "The resolution for the encoder",
        NTA_BasicType_Real64,
        1, // elementCount
        "", // constraints
        "0", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "radius",
      ParameterSpec(
        "The radius for the encoder",
        NTA_BasicType_Real64,
        1, // elementCount
        "", // constraints
        "0", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "minValue",
      ParameterSpec(
        "The minimum value for the input",
        NTA_BasicType_Real64,
        1, // elementCount
        "", // constraints
        "-1", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "maxValue",
      ParameterSpec(
        "The maximum value for the input",
        NTA_BasicType_Real64,
        1, // elementCount
        "", // constraints
        "-1", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "periodic",
      ParameterSpec(
        "Whether the encoder is periodic",
        NTA_BasicType_Bool,
        1, // elementCount
        "", // constraints
        "false", // defaultValue
        ParameterSpec::ReadWriteAccess));

    ns->parameters.add(
      "clipInput",
      ParameterSpec(
        "Whether to clip inputs if they're outside [minValue, maxValue]",
        NTA_BasicType_Bool,
        1, // elementCount
        "", // constraints
        "false", // defaultValue
        ParameterSpec::ReadWriteAccess));

    /* ----- outputs ----- */

    ns->outputs.add(
      "encoded",
      OutputSpec(
        "Encoded value",
        NTA_BasicType_Real32,
        0, // elementCount
        true, // isRegionLevel
        true // isDefaultOutput
        ));

    ns->outputs.add(
      "bucket",
      OutputSpec(
        "Bucket number for this sensedValue",
        NTA_BasicType_Int32,
        0, // elementCount
        true, // isRegionLevel
        false // isDefaultOutput
        ));

    return ns;
  }

  void
  ScalarSensor::getParameterFromBuffer(const std::string& name,
                                       Int64 index,
                                       IWriteBuffer& value)
  {
    if (name == "sensedValue")
    {
      value.write(sensedValue_);
    }
    else if (name == "n")
    {
      // Cast to UInt32 to avoid call resolution ambiguity on the write() method
      value.write((UInt32)encoder_->getOutputWidth());
    }
    else
    {
      NTA_THROW << "ScalarSensor::getParameter -- Unknown parameter " << name;
    }
  }

  void
  ScalarSensor::setParameterFromBuffer(const std::string& name,
                                       Int64 index,
                                       IReadBuffer& value)
  {
    if (name == "sensedValue")
    {
      value.read(sensedValue_);
    }
    else
    {
      NTA_THROW << "ScalarSensor::setParameter -- Unknown parameter " << name;
    }
  }

  void
  ScalarSensor::initialize()
  {
    encodedOutput_ = getOutput("encoded");
    bucketOutput_ = getOutput("bucket");
  }

  size_t
  ScalarSensor::getNodeOutputElementCount(const std::string& outputName)
  {
    if (outputName == "encoded")
    {
      return encoder_->getOutputWidth();
    }
    else if (outputName == "bucket")
    {
      return 1;
    }
    else
    {
      NTA_THROW << "ScalarSensor::getOutputSize -- unknown output " << outputName;
    }
  }

  std::string ScalarSensor::executeCommand(const std::vector<std::string>& args, Int64 index)
  {
    NTA_THROW << "ScalarSensor::executeCommand -- commands not supported";
  }

  void ScalarSensor::serialize(BundleIO& bundle)
  {
    NTA_THROW << "ScalarSensor::serialize -- Not implemented";
  }


  void ScalarSensor::deserialize(BundleIO& bundle)
  {
    NTA_THROW << "ScalarSensor::deserialize -- Not implemented";
  }


  void ScalarSensor::write(AnyPointer::Builder& anyProto) const
  {
    NTA_THROW << "ScalarSensor::write -- Not implemented";
  }


  void ScalarSensor::read(AnyPointer::Reader& anyProto)
  {
    NTA_THROW << "ScalarSensor::read -- Not implemented";
  }
}
