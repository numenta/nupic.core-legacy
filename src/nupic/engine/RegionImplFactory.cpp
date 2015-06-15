/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */


#include <stdexcept>
#include <nupic/engine/RegionImplFactory.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegisteredRegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/os/DynamicLibrary.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/engine/YAMLUtils.hpp>
#include <nupic/engine/TestNode.hpp>
#include <nupic/regions/VectorFileEffector.hpp>
#include <nupic/regions/VectorFileSensor.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/StringUtils.hpp>

// from http://stackoverflow.com/a/9096509/1781435
#define stringify(x)  #x
#define expand_and_stringify(x) stringify(x)

namespace nupic
{
  // Path from site-packages to packages that contain NuPIC Python regions
  static std::map<const std::string, std::set<const std::string>> py_regions;

  // Mappings for C++ regions
  static std::map<const std::string, GenericRegisteredRegionImpl*> cpp_regions;

  bool initializedRegions = false;

  // Allows the user to add custom regions
  void RegionImplFactory::registerPyRegion(const std::string module, const std::string className)
  {
    // Module hasn't been added yet
    if (py_regions.find(module) == py_regions.end())
      py_regions[module] = std::set<const std::string>();
        
    py_regions[module].insert(className);
  }

  void RegionImplFactory::registerCPPRegion(const std::string name, GenericRegisteredRegionImpl * wrapper)
  {
    cpp_regions[name] = wrapper;
  }

  class DynamicPythonLibrary
  {
    typedef void (*initPythonFunc)();
    typedef void (*finalizePythonFunc)();
    typedef void * (*createSpecFunc)(const char *, void **, const char *);
    typedef int (*destroySpecFunc)(const char *, const char *);
    typedef void * (*createPyNodeFunc)(const char *, void *, void *, void **, const char *);
    typedef void * (*deserializePyNodeFunc)(const char *, void *, void *, void *, const char *);
  public:
    DynamicPythonLibrary() :
      initPython_(nullptr),
      finalizePython_(nullptr),
      createSpec_(nullptr),
      destroySpec_(nullptr),
      createPyNode_(nullptr)
    {
      // To find the pynode plugin we need the nupic
      // installation directory.
#if defined(NTA_OS_WINDOWS)
      std::string command = "python -c \"import sys;import os;import nupic;sys.stdout.write(os.path.abspath(os.path.join(nupic.__file__, \"\"../..\"\")))\"";
#else
      std::string command = "python -c 'import sys;import os;import nupic;sys.stdout.write(os.path.abspath(os.path.join(nupic.__file__, \"../..\")))'";
#endif
      rootDir_ = OS::executeCommand(command);
      if (!Path::exists(rootDir_))
        NTA_THROW << "Unable to find NuPIC library in '" << rootDir_ << "'";
      
      
#if defined(NTA_OS_WINDOWS)
      const char * filename = "cpp_region.dll";
#else
      const char * filename = "libcpp_region.so";
#endif

      std::string libName = Path::join(rootDir_, "nupic", filename);

      if (!Path::exists(libName))
        NTA_THROW << "Unable to find library '" << libName << "'";

      std::string errorString;
      DynamicLibrary * p = 
        DynamicLibrary::load(libName, 
                             // export as LOCAL because we don't want
                             // the symbols to be globally visible; 
                             // But the python module that we load
                             // has to be able to access symbols from
                             // libpython.so; Since libpython.so is linked
                             // to the pynode shared library, it appears
                             // we have to make the pynode shared library
                             // symbols global. TODO: investigate
                             DynamicLibrary::GLOBAL| 
                             // Evaluate them NOW instead of LAZY to catch 
                             // errors up front, even though this takes
                             // a little longer to load the library. 
                             // However -- the current dependency chain
                             // PyNode->Region->RegionImplFactory apparently
                             // creates never-used dependencies on YAML
                             // so until this is resolved use LAZY
                             DynamicLibrary::LAZY,
                             errorString);
      NTA_CHECK(p) << "Unable to load the pynode library: " << errorString;

      pynodeLibrary_ = boost::shared_ptr<DynamicLibrary>(p);

      initPython_ = (initPythonFunc)pynodeLibrary_->getSymbol("NTA_initPython");
      NTA_CHECK(initPython_) << "Unable to find NTA_initPython symbol in " << filename;

      finalizePython_ = (finalizePythonFunc)pynodeLibrary_->getSymbol("NTA_finalizePython");
      NTA_CHECK(finalizePython_) << "Unable to find NTA_finalizePython symbol in " << filename;

      createPyNode_ = (createPyNodeFunc)pynodeLibrary_->getSymbol("NTA_createPyNode");
      NTA_CHECK(createPyNode_) << "Unable to find NTA_createPyNode symbol in " << filename;

      deserializePyNode_ = (deserializePyNodeFunc)pynodeLibrary_->getSymbol("NTA_deserializePyNode");
      NTA_CHECK(createPyNode_) << "Unable to find NTA_createPyNode symbol in " << filename;

      createSpec_ = (createSpecFunc)pynodeLibrary_->getSymbol("NTA_createSpec");
      NTA_CHECK(createSpec_) << "Unable to find NTA_createSpec symbol in " << filename;

      destroySpec_ = (destroySpecFunc)pynodeLibrary_->getSymbol("NTA_destroySpec");
      NTA_CHECK(destroySpec_) << "Unable to find NTA_destroySpec symbol in " << filename;

      (*initPython_)();
    }

    ~DynamicPythonLibrary()
    {
      //NTA_DEBUG << "In DynamicPythonLibrary Destructor";
      if (finalizePython_)
        finalizePython_();
    } 

    void * createSpec(std::string nodeType, void ** exception, std::string className)
    {
      //NTA_DEBUG << "RegionImplFactory::createSpec(" << nodeType << ")";
      return (*createSpec_)(nodeType.c_str(), exception, className.c_str());
    }

    int destroySpec(std::string nodeType, std::string& className)
    {
      NTA_INFO << "destroySpec(" << nodeType << ")";
      return (*destroySpec_)(nodeType.c_str(), className.c_str());
    }

    void * createPyNode(const std::string& nodeType, 
                        ValueMap * nodeParams,
                        Region * region,
                        void ** exception,
                        const std::string& className)
    {
      //NTA_DEBUG << "RegionImplFactory::createPyNode(" << nodeType << ")";
      return (*createPyNode_)(nodeType.c_str(),
                              reinterpret_cast<void *>(nodeParams),
                              reinterpret_cast<void*>(region),
                              exception,
                              className.c_str());

    }

    void * deserializePyNode(const std::string& nodeType, 
                             BundleIO* bundle,
                             Region * region, 
                             void ** exception,
                             const std::string& className)
    {
      //NTA_DEBUG << "RegionImplFactory::deserializePyNode(" << nodeType << ")";
      return (*deserializePyNode_)(nodeType.c_str(), 
                                   reinterpret_cast<void*>(bundle),
                                   reinterpret_cast<void*>(region), 
                                   exception,
                                   className.c_str());
    }

    const std::string& getRootDir() const
    {
      return rootDir_;
    }

  private:
    std::string rootDir_;
    boost::shared_ptr<DynamicLibrary> pynodeLibrary_;
    initPythonFunc initPython_;
    finalizePythonFunc finalizePython_;
    createSpecFunc createSpec_;
    destroySpecFunc destroySpec_;
    createPyNodeFunc createPyNode_;
    deserializePyNodeFunc deserializePyNode_;
  };

RegionImplFactory & RegionImplFactory::getInstance()
{
  static RegionImplFactory instance;

  // Initialize Regions
  if (!initializedRegions)
  {
    // Create C++ regions
    cpp_regions["TestNode"] = new RegisteredRegionImpl<TestNode>();
    cpp_regions["VectorFileEffector"] = new RegisteredRegionImpl<VectorFileEffector>();
    cpp_regions["VectorFileSensor"] = new RegisteredRegionImpl<VectorFileSensor>();

    // Create Python regions
    // nupic.regions
    RegionImplFactory::registerPyRegion("nupic.regions.AnomalyRegion", "AnomalyRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.CLAClassifierRegion", "CLAClassifierRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.ImageSensor", "ImageSensor");
    RegionImplFactory::registerPyRegion("nupic.regions.KNNAnomalyClassifierRegion", "KNNAnomalyClassifierRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.KNNClassifierRegion", "KNNClassifierRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.PCANode", "PCANode");
    RegionImplFactory::registerPyRegion("nupic.regions.PyRegion", "PyRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.RecordSensor", "RecordSensor");
    RegionImplFactory::registerPyRegion("nupic.regions.SPRegion", "SPRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.SVMClassifierNode", "SVMClassifierNode");
    RegionImplFactory::registerPyRegion("nupic.regions.TPRegion", "TPRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.TestNode", "TestNode");
    RegionImplFactory::registerPyRegion("nupic.regions.TestRegion", "TestRegion");
    RegionImplFactory::registerPyRegion("nupic.regions.UnimportableNode", "UnimportableNode");

    // nupic.regions.extra
    RegionImplFactory::registerPyRegion("nupic.regions.GaborNode2", "GaborNode2");

    initializedRegions = true;
  }

  return instance;
}

// This function creates either a NuPIC 2 or NuPIC 1 Python node 
static RegionImpl * createPyNode(DynamicPythonLibrary * pyLib, 
                                 const std::string & nodeType,
                                 ValueMap * nodeParams,
                                 Region * region)
{
  const std::string className(nodeType.c_str() + 3);
  for (auto pyr=py_regions.begin(); pyr!=py_regions.end(); pyr++)
  {
    const std::string module = pyr->first;
    std::set<const std::string> classes = pyr->second;

    // This module contains the class
    if (classes.find(className) != classes.end())
    {
      void * exception = nullptr;
      void * node = pyLib->createPyNode(module, nodeParams, region, &exception, className);
      if (node)
      {
        return static_cast<RegionImpl*>(node);
      }
    }
  }

  NTA_THROW << "Unable to create region " << region->getName() << " of type " << className;
  return nullptr;
}

// This function deserializes either a NuPIC 2 or NuPIC 1 Python node 
static RegionImpl * deserializePyNode(DynamicPythonLibrary * pyLib, 
                                      const std::string & nodeType,
                                      BundleIO & bundle,
                                      Region * region)
{
  const std::string className(nodeType.c_str() + 3);
  for (auto pyr=py_regions.begin(); pyr!=py_regions.end(); pyr++)
  {
    const std::string module = pyr->first;
    std::set<const std::string> classes = pyr->second;

    // This module contains the class
    if (classes.find(className) != classes.end())
    {
      void * exception = nullptr;
      void * node = pyLib->deserializePyNode(module, &bundle, region, &exception, className);
      if (node)
      {
        return static_cast<RegionImpl*>(node);
      }
    }
  }

  NTA_THROW << "Unable to deserialize region " << region->getName() << " of type " << className;
  return nullptr;



}

RegionImpl* RegionImplFactory::createRegionImpl(const std::string nodeType, 
                                                const std::string nodeParams,
                                                Region* region)
{

  RegionImpl *mn = nullptr;
  Spec *ns = getSpec(nodeType);
  ValueMap vm = YAMLUtils::toValueMap(
    nodeParams.c_str(), 
    ns->parameters, 
    nodeType, 
    region->getName());
    
  if (cpp_regions.find(nodeType) != cpp_regions.end())
  {
    mn = cpp_regions[nodeType]->createRegionImpl(vm, region);
  }
  else if ((nodeType.find(std::string("py.")) == 0))
  {
    if (!pyLib_)
      pyLib_ = boost::shared_ptr<DynamicPythonLibrary>(new DynamicPythonLibrary());
    
    mn = createPyNode(pyLib_.get(), nodeType, &vm, region);
  } else
  {
    NTA_THROW << "Unsupported node type '" << nodeType << "'";
  }

  return mn;

}

RegionImpl* RegionImplFactory::deserializeRegionImpl(const std::string nodeType, 
                                                     BundleIO& bundle,
                                                     Region* region)
{

  RegionImpl *mn = nullptr;

  if (cpp_regions.find(nodeType) != cpp_regions.end())
  {
    mn = cpp_regions[nodeType]->deserializeRegionImpl(bundle, region);
  }
  else if (StringUtils::startsWith(nodeType, "py."))
  {
    if (!pyLib_)
      pyLib_ = boost::shared_ptr<DynamicPythonLibrary>(new DynamicPythonLibrary());
    
    mn = deserializePyNode(pyLib_.get(), nodeType, bundle, region);
  } else
  {
    NTA_THROW << "Unsupported node type '" << nodeType << "'";
  }
  return mn;

}

// This function returns the node spec of a NuPIC 2 or NuPIC 1 Python node 
static Spec * getPySpec(DynamicPythonLibrary * pyLib,
                                const std::string & nodeType)
{
  const std::string className(nodeType.c_str() + 3);
  for (auto pyr=py_regions.begin(); pyr!=py_regions.end(); pyr++)
  {
    const std::string module = pyr->first;
    std::set<const std::string> classes = pyr->second;

    // This module contains the class
    if (classes.find(className) != classes.end())
    {
      void * exception = nullptr;
      void * ns = pyLib->createSpec(module, &exception, className);
      if (ns)
      {
        return (Spec *)ns;
      }
    }
  }

  NTA_THROW << "Matching Python module for " << className << " not found.";
}

Spec * RegionImplFactory::getSpec(const std::string nodeType)
{
  std::map<std::string, Spec*>::iterator it;
  // return from cache if we already have it
  it = nodespecCache_.find(nodeType);
  if (it != nodespecCache_.end())
    return it->second;

  // grab the nodespec and cache it
  // one entry per supported node type
  Spec * ns = nullptr;
  if (cpp_regions.find(nodeType) != cpp_regions.end())
  {
    ns = cpp_regions[nodeType]->createSpec();
  }
  else if (nodeType.find(std::string("py.")) == 0)
  {
    if (!pyLib_)
      pyLib_ = boost::shared_ptr<DynamicPythonLibrary>(new DynamicPythonLibrary());

    ns = getPySpec(pyLib_.get(), nodeType);
  } 
  else 
  {
    NTA_THROW << "getSpec() -- Unsupported node type '" << nodeType << "'";
  }

  if (!ns)
    NTA_THROW << "Unable to get node spec for: " << nodeType;

  nodespecCache_[nodeType] = ns;
  return ns;
}
    
void RegionImplFactory::cleanup()
{
  std::map<std::string, Spec*>::iterator ns;
  // destroy all nodespecs
  for (ns = nodespecCache_.begin(); ns != nodespecCache_.end(); ns++)
  {
    assert(ns->second != nullptr);
    // PyNode node specs are destroyed by the C++ PyNode
    if (ns->first.substr(0, 3) == "py.")
    {
      std::string noClass = "";
      pyLib_->destroySpec(ns->first, noClass);
    }
    else
    {
      delete ns->second;
    }

    ns->second = nullptr;
  }

  nodespecCache_.clear();

  // destroy all RegisteredRegionImpls
  for (auto rri = cpp_regions.begin(); rri != cpp_regions.end(); rri++)
  {
    NTA_ASSERT(rri->second != nullptr);
    delete rri->second;
    rri->second = nullptr;
  }

  cpp_regions.clear();

  // Never release the Python dynamic library!
  // This is due to cleanup issues of Python itself
  // See: http://docs.python.org/c-api/init.html#Py_Finalize
  //pyLib_.reset();
}

}

