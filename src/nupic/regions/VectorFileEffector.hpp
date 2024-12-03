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
 * Declarations for VectorFileEffector class
 */

//----------------------------------------------------------------------

#ifndef NTA_VECTOR_FILE_EFFECTOR_HPP
#define NTA_VECTOR_FILE_EFFECTOR_HPP

//----------------------------------------------------------------------

#include <capnp/any.h>

#include <nupic/engine/RegionImpl.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/os/FStream.hpp>
#include <nupic/types/Types.h>

namespace nupic {

class ValueMap;

/**
 *  VectorFileEffector is a node that takes its input vectors and
 *  writes them sequentially to a file.
 *
 *  The current input vector is written (but not flushed) to the file
 *  each time the effector's compute() method is called.
 *
 *  The file format for the file is a space-separated list of numbers, with
 *  one vector per line:
 *
 *        e11 e12 e13 ... e1N
 *        e21 e22 e23 ... e2N
 *           :
 *        eM1 eM2 eM3 ... eMN
 *
 *  VectorFileEffector implements the execute() commands as defined in the
 *  nodeSpec.
 *
 */
class VectorFileEffector : public RegionImpl {
public:
  static Spec *createSpec();
  size_t getNodeOutputElementCount(const std::string &outputName) override;
  void getParameterFromBuffer(const std::string &name, Int64 index,
                              IWriteBuffer &value) override;

  void setParameterFromBuffer(const std::string &name, Int64 index,
                              IReadBuffer &value) override;

  void setParameterString(const std::string &name, Int64 index,
                          const std::string &s) override;
  std::string getParameterString(const std::string &name, Int64 index) override;

  void initialize() override;

  VectorFileEffector(const ValueMap &params, Region *region);

  VectorFileEffector(BundleIO &bundle, Region *region);

  VectorFileEffector(capnp::AnyPointer::Reader &proto, Region *region);

  virtual ~VectorFileEffector();

  // ---
  /// Serialize state to bundle
  // ---
  virtual void serialize(BundleIO &bundle) override;

  // ---
  /// De-serialize state from bundle
  // ---
  virtual void deserialize(BundleIO &bundle) override;

  using RegionImpl::write;
  virtual void write(capnp::AnyPointer::Builder &anyProto) const override;

  using RegionImpl::read;
  virtual void read(capnp::AnyPointer::Reader &anyProto) override;

  void compute() override;

  virtual std::string executeCommand(const std::vector<std::string> &args,
                                     Int64 index) override;

private:
  void closeFile();
  void openFile(const std::string &filename);

  ArrayRef dataIn_;
  std::string filename_;     // Name of the output file
  nupic::OFStream *outFile_; // Handle to current file

  /// Disable unsupported default constructors
  VectorFileEffector(const VectorFileEffector &);
  VectorFileEffector &operator=(const VectorFileEffector &);

}; // end class VectorFileEffector

//----------------------------------------------------------------------

} // namespace nupic

#endif // NTA_VECTOR_FILE_EFFECTOR_HPP
