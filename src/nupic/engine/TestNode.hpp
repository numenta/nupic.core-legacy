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

#ifndef NTA_TESTNODE_HPP
#define NTA_TESTNODE_HPP

#include <string>
#include <vector>

// Workaround windows.h collision:
// https://github.com/sandstorm-io/capnproto/issues/213
#undef VOID
#include <capnp/any.h>

#include <nupic/engine/RegionImpl.hpp>
#include <nupic/ntypes/Value.hpp>

namespace nupic {

/*
 * TestNode is does simple computations of inputs->outputs
 * inputs and outputs are Real64 arrays
 *
 * delta is a parameter used for the computation. defaults to 1
 *
 * Size of each node output is given by the outputSize parameter (cg)
 * which defaults to 2 and cannot be less than 1. (parameter not yet
 implemented)
 *
 * Here is the totally lame "computation"
 * output[0] = number of inputs to this baby node + current iteration number (0
 for first compute)
 * output[1] = baby node num + sum of inputs to this baby node
 * output[2] = baby node num + sum of inputs + (delta)
 * output[3] = baby node num + sum of inputs + (2*delta)
 * ...
 * output[n] = baby node num + sum of inputs + ((n-1) * delta)

 * It can act as a sensor if no inputs are connected (sum of inputs = 0)
 */

class BundleIO;

class TestNode : public RegionImpl {
public:
  typedef void (*computeCallbackFunc)(const std::string &);
  TestNode(const ValueMap &params, Region *region);
  TestNode(BundleIO &bundle, Region *region);
  TestNode(capnp::AnyPointer::Reader &proto, Region *region);
  virtual ~TestNode();

  /* -----------  Required RegionImpl Interface methods ------- */

  // Used by RegionImplFactory to create and cache
  // a nodespec. Ownership is transferred to the caller.
  static Spec *createSpec();

  std::string getNodeType() { return "TestNode"; };
  void compute() override;
  std::string executeCommand(const std::vector<std::string> &args,
                             Int64 index) override;

  size_t getNodeOutputElementCount(const std::string &outputName) override;
  void getParameterFromBuffer(const std::string &name, Int64 index,
                              IWriteBuffer &value) override;
  void setParameterFromBuffer(const std::string &name, Int64 index,
                              IReadBuffer &value) override;

  void initialize() override;

  void serialize(BundleIO &bundle) override;
  void deserialize(BundleIO &bundle) override;

  using RegionImpl::write;
  virtual void write(capnp::AnyPointer::Builder &anyProto) const override;

  using RegionImpl::read;
  virtual void read(capnp::AnyPointer::Reader &anyProto) override;

  /* -----------  Optional RegionImpl Interface methods ------- */

  size_t getParameterArrayCount(const std::string &name, Int64 index) override;

  // Override for Real64 only
  // We choose Real64 in the test node to preserve precision. All other type
  // go through read/write buffer serialization, and floating point values may
  // get truncated in the conversion to/from ascii.
  Real64 getParameterReal64(const std::string &name, Int64 index) override;
  void setParameterReal64(const std::string &name, Int64 index,
                          Real64 value) override;

  bool isParameterShared(const std::string &name) override;

private:
  TestNode();

  // parameters
  // cgs parameters for parameter testing
  Int32 int32Param_;
  UInt32 uint32Param_;
  Int64 int64Param_;
  UInt64 uint64Param_;
  Real32 real32Param_;
  Real64 real64Param_;
  bool boolParam_;
  std::string stringParam_;
  computeCallbackFunc computeCallback_;

  std::vector<Real32> real32ArrayParam_;
  std::vector<Int64> int64ArrayParam_;
  std::vector<bool> boolArrayParam_;

  // read-only count of iterations since initialization
  UInt64 iter_;

  // Constructor param specifying per-node output size
  UInt32 outputElementCount_;

  // parameter used for computation
  Int64 delta_;

  // cloning parameters
  std::vector<UInt32> unclonedParam_;
  bool shouldCloneParam_;
  std::vector<UInt32> possiblyUnclonedParam_;
  std::vector<std::vector<Int64>> unclonedInt64ArrayParam_;

  /* ----- cached info from region ----- */
  size_t nodeCount_;

  // Input/output buffers for the whole region
  const Input *bottomUpIn_;
  const Output *bottomUpOut_;
};
} // namespace nupic

#endif // NTA_TESTNODE_HPP
