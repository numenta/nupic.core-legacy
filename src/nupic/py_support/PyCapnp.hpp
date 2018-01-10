/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

// This file contains utility functions for converting from pycapnp schema to
// compiled in schema and vice versa.
// It requires linking to both libcapnp and libcapnpc.

#ifndef NTA_PY_CAPNP_HPP
#define NTA_PY_CAPNP_HPP

#include <stdexcept>  // for std::logic_error

#include <Python.h>

#if !CAPNP_LITE

  #include <capnp/any.h>
  #include <capnp/dynamic.h>
  #include <capnp/message.h>
  #include <capnp/schema-parser.h>
#endif // !CAPNP_LITE

#include <nupic/types/Serializable.hpp>



namespace nupic
{

  class PyCapnpHelper
  {
  public:
    /**
     * Serialize the given nupic::Serializable-based instance, returning a capnp
     * byte buffer as python byte string.
     *
     * :param obj: The Serializable object
     *
     * :returns: capnp byte buffer encoded as python byte string.
     *
     * :example: PyObject* pyBytes = PyCapnpHelper::writeAsPyBytes(*netPtr);
     */
    template<class MessageType>
    static PyObject* writeAsPyBytes(const nupic::Serializable<MessageType>& obj)
    {
    #if !CAPNP_LITE
      capnp::MallocMessageBuilder message;
      typename MessageType::Builder proto = message.initRoot<MessageType>();

      obj.write(proto);

      // Extract message data and convert to Python byte object
      kj::Array<capnp::word> array = capnp::messageToFlatArray(message); // copy
      kj::ArrayPtr<kj::byte> byteArray = array.asBytes();
      PyObject* result = PyString_FromStringAndSize(
        (const char*)byteArray.begin(),
        byteArray.size()); // copy
      return result;
    #else
      throw std::logic_error(
        "PyCapnpHelper::writeAsPyBytes is not implemented when "
        "compiled with CAPNP_LITE=1.");
    #endif
    }

    /**
     * Initialize the given nupic::Serializable-based instance from the given
     * Capnp message reader.
     *
     * :param pyBytes: The Serializable object
     *
     * :returns: capnp byte buffer encoded as python byte string.
     *
     * :example: PyCapnpHelper::initFromPyBytes(network, pyBytes);
     */
    template<class MessageType>
    static void initFromPyBytes(
      nupic::Serializable<MessageType>& obj,
      const PyObject* pyBytes)
    {
    #if !CAPNP_LITE
      const char* srcBytes = nullptr;
      Py_ssize_t srcNumBytes = 0;

      // NOTE: srcBytes will be set to point to the internal buffer inside
      // pyRegionProtoBytes'
      PyString_AsStringAndSize(const_cast<PyObject*>(pyBytes),
                               const_cast<char**>(&srcBytes),
                               &srcNumBytes);

      if (srcNumBytes % sizeof(capnp::word) != 0)
      {
        throw std::logic_error(
          "PyCapnpHelper.initFromPyBytes input length must be a multiple of "
          "capnp::word.");
      }
      const int srcNumWords = srcNumBytes / sizeof(capnp::word);

      // Ensure alignment on capnp::word boundary; the buffer inside PyObject appears
      // to be unaligned on capnp::word boundary.
      kj::Array<capnp::word> array = kj::heapArray<capnp::word>(srcNumWords);
      memcpy(array.asBytes().begin(), srcBytes, srcNumBytes); // copy

      capnp::FlatArrayMessageReader reader(array.asPtr()); // copy ?
      typename MessageType::Reader proto = reader.getRoot<MessageType>();
      obj.read(proto);
    #else
      throw std::logic_error(
        "PyCapnpHelper::initFromPyBytes is not implemented when "
        "compiled with CAPNP_LITE=1.");

    #endif
    }

  }; // class PyCapnpHelper

}  // namespace nupic


#endif  // NTA_PY_CAPNP_HPP
