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
 * Implementation for VectorFileEffector class
 */

#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>

#include <capnp/any.h>

#include <nupic/engine/Region.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/os/FStream.hpp>
#include <nupic/regions/VectorFileEffector.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

VectorFileEffector::VectorFileEffector(const ValueMap &params, Region *region)
    : RegionImpl(region), dataIn_(NTA_BasicType_Real32), filename_(""),
      outFile_(nullptr) {
  if (params.contains("outputFile"))
    filename_ = *params.getString("outputFile");
  else
    filename_ = "";
}

VectorFileEffector::VectorFileEffector(BundleIO &bundle, Region *region)
    : RegionImpl(region), dataIn_(NTA_BasicType_Real32), filename_(""),
      outFile_(nullptr) {}

VectorFileEffector::VectorFileEffector(capnp::AnyPointer::Reader &proto,
                                       Region *region)
    : RegionImpl(region), dataIn_(NTA_BasicType_Real32), filename_(""),
      outFile_(nullptr) {
  read(proto);
}

VectorFileEffector::~VectorFileEffector() { closeFile(); }

void VectorFileEffector::initialize() {
  NTA_CHECK(region_ != nullptr);
  // We have no outputs or parameters; just need our input.
  dataIn_ = region_->getInputData("dataIn");

  if (dataIn_.getCount() == 0) {
    NTA_THROW << "VectorFileEffector::init - no input found\n";
  }
}

void VectorFileEffector::compute() {

  // It's not necessarily an error to have no inputs. In this case we just
  // return
  if (dataIn_.getCount() == 0)
    return;

  // Don't write if there is no open file.
  if (outFile_ == nullptr) {
    NTA_WARN
        << "VectorFileEffector compute() called, but there is no open file";
    return;
  }

  // Ensure we can write to it
  if (outFile_->fail()) {
    NTA_THROW << "VectorFileEffector: There was an error writing to the file "
              << filename_.c_str() << "\n";
  }

  Real *inputVec = (Real *)(dataIn_.getBuffer());
  NTA_CHECK(inputVec != nullptr);
  OFStream &outFile = *outFile_;
  for (Size offset = 0; offset < dataIn_.getCount(); ++offset) {
    // TBD -- could be very inefficient to do one at a time
    outFile << inputVec[offset] << " ";
  }
  outFile << "\n";
}

void VectorFileEffector::closeFile() {
  if (outFile_) {
    outFile_->close();
    outFile_ = nullptr;
    filename_ = "";
  }
}

void VectorFileEffector::openFile(const std::string &filename) {

  if (outFile_ && !outFile_->fail())
    closeFile();
  if (filename == "")
    return;

  outFile_ = new OFStream(filename.c_str(), std::ios::app);
  if (outFile_->fail()) {
    delete outFile_;
    outFile_ = nullptr;
    NTA_THROW
        << "VectorFileEffector::openFile -- unable to create or open file: "
        << filename.c_str();
  }
  filename_ = filename;
}

void VectorFileEffector::setParameterString(const std::string &paramName,
                                            Int64 index, const std::string &s) {

  if (paramName == "outputFile") {
    if (s == filename_ && outFile_)
      return; // already set
    if (outFile_)
      closeFile();
    openFile(s);
  } else {
    NTA_THROW << "VectorFileEffector -- Unknown string parameter " << paramName;
  }
}

std::string VectorFileEffector::getParameterString(const std::string &paramName,
                                                   Int64 index) {
  if (paramName == "outputFile") {
    return filename_;
  } else {
    NTA_THROW << "VectorFileEffector -- unknown parameter " << paramName;
  }
}

std::string
VectorFileEffector::executeCommand(const std::vector<std::string> &args,
                                   Int64 index) {
  NTA_CHECK(args.size() > 0);
  // Process the flushFile command
  if (args[0] == "flushFile") {
    // Ensure we have a valid file before flushing, otherwise fail silently.
    if (!((outFile_ == nullptr) || (outFile_->fail()))) {
      outFile_->flush();
    }
  } else if (args[0] == "closeFile") {
    closeFile();
  } else if (args[0] == "echo") {
    // Ensure we have a valid file before flushing, otherwise fail silently.
    if ((outFile_ == nullptr) || (outFile_->fail())) {
      NTA_THROW << "VectorFileEffector: echo command failed because there is "
                   "no file open";
    }

    for (size_t i = 1; i < args.size(); i++) {
      *outFile_ << args[i];
    }
    *outFile_ << "\n";
  } else {
    NTA_THROW << "VectorFileEffector: Unknown execute '" << args[0] << "'";
  }

  return "";
}

Spec *VectorFileEffector::createSpec() {

  auto ns = new Spec;
  ns->description =
      "VectorFileEffector is a node that simply writes its\n"
      "input vectors to a text file. The target filename is specified\n"
      "using the 'outputFile' parameter at run time. On each\n"
      "compute, the current input vector is written (but not flushed)\n"
      "to the file.\n";

  ns->inputs.add("dataIn",
                 InputSpec("Data to be written to file", NTA_BasicType_Real32,
                           0,     // count
                           false, // required?
                           false, // isRegionLevel
                           true   // isDefaultInput
                           ));

  ns->parameters.add("outputFile",
                     ParameterSpec("Writes output vectors to this file on each "
                                   "compute. Will append to any\n"
                                   "existing data in the file. This parameter "
                                   "must be set at runtime before\n"
                                   "the first compute is called. Throws an "
                                   "exception if it is not set or\n"
                                   "the file cannot be written to.\n",
                                   NTA_BasicType_Byte,
                                   0,  // elementCount
                                   "", // constraints
                                   "", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->commands.add("flushFile", CommandSpec("Flush file data to disk"));

  ns->commands.add("closeFile",
                   CommandSpec("Close the current file, if open."));

  return ns;
}

size_t
VectorFileEffector::getNodeOutputElementCount(const std::string &outputName) {
  NTA_THROW
      << "VectorFileEffector::getNodeOutputElementCount -- unknown output '"
      << outputName << "'";
}

void VectorFileEffector::getParameterFromBuffer(const std::string &name,
                                                Int64 index,
                                                IWriteBuffer &value) {
  NTA_THROW << "VectorFileEffector -- unknown parameter '" << name << "'";
}

void VectorFileEffector::setParameterFromBuffer(const std::string &name,
                                                Int64 index,
                                                IReadBuffer &value) {
  NTA_THROW << "VectorFileEffector -- unknown parameter '" << name << "'";
}

void VectorFileEffector::serialize(BundleIO &bundle) { return; }

void VectorFileEffector::deserialize(BundleIO &bundle) { return; }

void VectorFileEffector::write(capnp::AnyPointer::Builder &anyProto) const {
  return;
}

void VectorFileEffector::read(capnp::AnyPointer::Reader &anyProto) { return; }

} // namespace nupic
