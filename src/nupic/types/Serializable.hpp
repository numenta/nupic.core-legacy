/*
 * Copyright 2015 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Definitions for the base Serializable class in C++
 */

#ifndef NTA_serializable_HPP
#define NTA_serializable_HPP

#include <iostream>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

namespace nupic {

/**
 * Base Serializable class that any serializable class
 * should inherit from.
 */
template <class ProtoT> class Serializable {
public:
  void write(std::ostream &stream) const {
    capnp::MallocMessageBuilder message;
    typename ProtoT::Builder proto = message.initRoot<ProtoT>();
    write(proto);

    kj::std::StdOutputStream out(stream);
    capnp::writeMessage(out, message);
  }

  void read(std::istream &stream) {
    kj::std::StdInputStream in(stream);
    capnp::ReaderOptions options;
    options.traversalLimitInWords = kj::maxValue; // Don't limit.
    capnp::InputStreamMessageReader message(in, options);
    typename ProtoT::Reader proto = message.getRoot<ProtoT>();
    read(proto);
  }

  virtual void write(typename ProtoT::Builder &proto) const = 0;
  virtual void read(typename ProtoT::Reader &proto) = 0;

  virtual ~Serializable() {}
};

} // end namespace nupic
#endif // NTA_serializable_HPP
