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

/** @file */

//----------------------------------------------------------------------

#include <nupic/os/DynamicLibrary.hpp>
#include <nupic/os/Path.hpp>
#include <sstream>
#include <iostream>

//----------------------------------------------------------------------

namespace nupic
{

  DynamicLibrary::DynamicLibrary(void * handle) : handle_(handle)
  {
  }

  DynamicLibrary::~DynamicLibrary()
  {
    #if defined(NTA_OS_WINDOWS)
      ::FreeLibrary((HMODULE)handle_);
    #else
      ::dlclose(handle_);
    #endif
  }

  DynamicLibrary * DynamicLibrary::load(const std::string & name, std::string &errorString)
  {
    #if defined(NTA_OS_WINDOWS)
      return load(name, 0, errorString);
    #else
      // LOCAL/NOW make more sense. In NuPIC 2 we currently need GLOBAL/LAZY
      // See comments in RegionImplFactory.cpp
      // return load(name, LOCAL | NOW, errorString);
      return load(name, GLOBAL | LAZY, errorString);
    #endif
  }

  DynamicLibrary * DynamicLibrary::load(const std::string & name, UInt32 mode, std::string & errorString)
  {
    if (name.empty()) 
    {
      errorString = "Empty path.";
      return nullptr;
    }

    //if (!Path::exists(name))
    //{
    //  errorString = "Dynamic library doesn't exist.";
    //  return NULL;
    //}
    
    void * handle = nullptr;
  
    #if defined(NTA_OS_WINDOWS)
      #if !defined(NTA_COMPILER_GNU)
        mode; // ignore on Windows
      #endif
      handle = ::LoadLibraryA(name.c_str());
      if (handle == NULL)
      {
        DWORD errorCode = ::GetLastError();
        std::stringstream ss;
        ss << std::string("LoadLibrary(") << name 
           << std::string(") Failed. errorCode: ") 
           << errorCode; 
        errorString = ss.str();
        return NULL;
      }
    #else
      handle = ::dlopen(name.c_str(), mode);
      if (!handle) 
      {
        std::string dlErrorString;
        const char *zErrorString = ::dlerror();
        if (zErrorString)
          dlErrorString = zErrorString;
        errorString += "Failed to load \"" + name + '"';
        if(dlErrorString.size())
          errorString += ": " + dlErrorString;
        return nullptr;
      }

    #endif
    return new DynamicLibrary(handle);

  }

  void * DynamicLibrary::getSymbol(const std::string & symbol)
  {    
    #if defined(NTA_OS_WINDOWS)
      return (void*)::GetProcAddress((HMODULE)handle_, symbol.c_str());
    #else
      return ::dlsym(handle_, symbol.c_str());
    #endif
  }
}
