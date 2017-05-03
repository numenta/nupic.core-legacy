/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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


#ifndef NTA_PY_REGION_HPP
#define NTA_PY_REGION_HPP

#include <nupic/py_support/PyArray.hpp>

#include <string>
#include <vector>
#include <set>

#include <capnp/any.h>

#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Value.hpp>

namespace nupic
{
  struct Spec;

  class PyRegion : public RegionImpl
  {
    typedef std::map<std::string, Spec> SpecMap;
  public:
    // Used by RegionImplFactory to create and cache a nodespec
    static Spec * createSpec(const char * nodeType, const char* className="");

    // Used by RegionImplFactory to destroy a node spec when clearing its cache
    static void destroySpec(const char * nodeType, const char* className="");

    PyRegion(const char * module, const ValueMap & nodeParams, Region * region, const char* className="");
    PyRegion(const char * module, BundleIO& bundle, Region * region, const char* className="");
    PyRegion(const char * module, capnp::AnyPointer::Reader& proto, Region * region, const char* className="");
    virtual ~PyRegion();

    // DynamicPythonLibrary functions. Originally used NTA_EXPORT
    static void NTA_initPython();
    static void NTA_finalizePython();
    static void * NTA_createPyNode(const char * module, void * nodeParams,
      void * region, void ** exception, const char* className="");
    static void * NTA_deserializePyNode(const char * module, void * bundle,
      void * region, void ** exception, const char* className="");
    static void * NTA_deserializePyNodeProto(const char * module, void * proto,
      void * region, void ** exception, const char* className="");
    static const char * NTA_getLastError();
    static void * NTA_createSpec(const char * nodeType, void ** exception, const char* className="");
    static int NTA_destroySpec(const char * nodeType, const char* className="");

    // Manual serialization methods. Current recommended method.
    void serialize(BundleIO& bundle) override;
    void deserialize(BundleIO& bundle) override;

    // Capnp serialization methods - not yet implemented for PyRegions. This
    // method will replace serialize/deserialize once fully implemented
    // throughout codebase.
    using RegionImpl::write;
    /**
     * Serialize instance to the given message builder
     *
     * :param proto: PyRegionProto builder masquerading as AnyPointer builder
     */
    void write(capnp::AnyPointer::Builder& proto) const override;

    using RegionImpl::read;
    /**
     * Initialize instance from the given message reader
     *
     * :param proto: PyRegionProto reader masquerading as AnyPointer reader
     */
    void read(capnp::AnyPointer::Reader& proto) override;

    const Spec & getSpec();

    static void createSpec(const char * nodeType, Spec & ns, const char* className="");

    // RegionImpl interface

    size_t getNodeOutputElementCount(const std::string& outputName) override;
    void getParameterFromBuffer(const std::string& name, Int64 index, IWriteBuffer& value) override;
    void setParameterFromBuffer(const std::string& name, Int64 index, IReadBuffer& value) override;

    void initialize() override;
    void compute() override;
    std::string executeCommand(
        const std::vector<std::string>& args, Int64 index) override;

    size_t getParameterArrayCount(const std::string& name, Int64 index)
        override;

    virtual Byte getParameterByte(const std::string& name, Int64 index);
    virtual Int32 getParameterInt32(const std::string& name, Int64 index)
        override;
    virtual UInt32 getParameterUInt32(const std::string& name, Int64 index)
        override;
    virtual Int64 getParameterInt64(const std::string& name, Int64 index)
        override;
    virtual UInt64 getParameterUInt64(const std::string& name, Int64 index)
        override;
    virtual Real32 getParameterReal32(const std::string& name, Int64 index)
        override;
    virtual Real64 getParameterReal64(const std::string& name, Int64 index)
        override;
    virtual Handle getParameterHandle(const std::string& name, Int64 index)
        override;
    virtual bool getParameterBool(const std::string& name, Int64 index)
        override;
    virtual std::string getParameterString(
        const std::string& name, Int64 index) override;

    virtual void setParameterByte(
        const std::string& name, Int64 index, Byte value);
    virtual void setParameterInt32(
        const std::string& name, Int64 index, Int32 value) override;
    virtual void setParameterUInt32(
        const std::string& name, Int64 index, UInt32 value) override;
    virtual void setParameterInt64(
        const std::string& name, Int64 index, Int64 value) override;
    virtual void setParameterUInt64(
        const std::string& name, Int64 index, UInt64 value) override;
    virtual void setParameterReal32(
        const std::string& name, Int64 index, Real32 value) override;
    virtual void setParameterReal64(
        const std::string& name, Int64 index, Real64 value) override;
    virtual void setParameterHandle(
        const std::string& name, Int64 index, Handle value) override;
    virtual void setParameterBool(
        const std::string& name, Int64 index, bool value) override;
    virtual void setParameterString(
        const std::string& name, Int64 index, const std::string& value)
        override;

    virtual void getParameterArray(
        const std::string& name, Int64 index, Array & array) override;
    virtual void setParameterArray(
        const std::string& name, Int64 index, const Array & array) override;

    // Helper methods
    template <typename T, typename PyT>
    T getParameterT(const std::string & name, Int64 index);

    template <typename T, typename PyT>
    void setParameterT(const std::string & name, Int64 index, T value);

  private:
    PyRegion();
    PyRegion(const Region &);

  private:
    static SpecMap specs_;
    std::string module_;
    std::string className_;
    py::Instance node_;
    std::set<boost::shared_ptr<PyArray<UInt64> > > splitterMaps_;
    // pointers rather than objects because Array doesnt
    // have a default constructor
    std::map<std::string, Array*> inputArrays_;
  };
}

#endif // NTA_PY_REGION_HPP
