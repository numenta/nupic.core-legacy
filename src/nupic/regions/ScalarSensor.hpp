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
 * Defines the ScalarSensor
 */

#ifndef NTA_SCALAR_SENSOR_HPP
#define NTA_SCALAR_SENSOR_HPP

#include <string>
#include <vector>

#include <nupic/encoders/ScalarEncoder.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/ntypes/Value.hpp>

namespace nupic {
/**
 * A network region that encapsulates the ScalarEncoder.
 *
 * @b Description
 * A ScalarSensor encapsulates ScalarEncoders, connecting them to the Network
 * API. As a network runs, the client will specify new encoder inputs by
 * setting the "sensedValue" parameter. On each compute, the ScalarSensor will
 * encode its "sensedValue" to output.
 */
class ScalarSensor : public RegionImpl {
public:
  ScalarSensor(const ValueMap &params, Region *region);
  ScalarSensor(BundleIO &bundle, Region *region);
  ScalarSensor();
  virtual ~ScalarSensor() override;

  static Spec *createSpec();

  virtual void getParameterFromBuffer(const std::string &name, Int64 index,
                                      IWriteBuffer &value) override;
  virtual void setParameterFromBuffer(const std::string &name, Int64 index,
                                      IReadBuffer &value) override;
  virtual void initialize() override;

  virtual void serialize(BundleIO &bundle) override;
  virtual void deserialize(BundleIO &bundle) override;


  void compute() override;
  virtual std::string executeCommand(const std::vector<std::string> &args,
                                     Int64 index) override;

  virtual size_t
  getNodeOutputElementCount(const std::string &outputName) override;

private:
  Real64 sensedValue_;
  ScalarEncoderBase *encoder_;
  Output *encodedOutput_;
  Output *bucketOutput_;
};
} // namespace nupic

#endif // NTA_SCALAR_SENSOR_HPP
