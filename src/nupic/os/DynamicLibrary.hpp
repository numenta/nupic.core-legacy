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

/** @file */

#ifndef NTA_DYNAMIC_LIBRARY_HPP
#define NTA_DYNAMIC_LIBRARY_HPP

//----------------------------------------------------------------------

#if defined(NTA_OS_WINDOWS)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <nupic/types/Types.hpp>
#include <string>

//----------------------------------------------------------------------

namespace nupic {
/**
  * @b Responsibility:
  *  1. Proivde a cross-platform dynamic library load/unload/getSymbol
  functionality
  *
  * @b Rationale:
  *  Numenta needs to load code dynamically on multiple platforms. It makes
  sense to
  *  encapsulate this core capability in a nice object-oriented C++ class.

  * @b Resource/Ownerships:
  *  1. An opaque library handle (released automatically by destructor)
  *
  * @b Invariants:
  *  1. handle_ is never NULL after construction. This invariant is guarantueed
  by
  *     the class design. The handle_ variable is private. The constructor that
  *     sets it is private. The load() factory method is the only method that
  invokes
  *     this constructor. The user has no chance to mess things up. The
  destructor
  *     cleans up by unloading the library.
  *
  * @b Notes:
  *  The load() static factory method is overloaded to provide default loading
  *  or loading based on an integer flag. The reason I didn't use a default
  *  argument is that the flag is an implememntation detail. An alternative
  *  approach is to define an enum with various flags that will be
  *  platform-independent and will be interpreted in the specific
  implementation.
  *
  *  The error handling strategy is to return error NULLs and not to throw
  exceptions.
  *  The reason is that it is a very generic low-level class that should not be
  aware
  *  and depend on the runtime's error handling policy. It may be used in
  different
  *  contexts like tools and utilities that may utilize a different error
  handling
  *  strategy. It is also a common idiom to return NULL from a failed factory
  method.
  *
  */
class DynamicLibrary {
public:
  enum Mode {
#if defined(NTA_OS_WINDOWS)
    LAZY,
    GLOBAL,
    LOCAL,
    NOW
#else
    LAZY = RTLD_LAZY,
    GLOBAL = RTLD_GLOBAL,
    LOCAL = RTLD_LOCAL,
    NOW = RTLD_NOW
#endif
  };

  /**
   * Loads a dynamic library file, stores the handle in a heap-allocated
   * DynamicLibrary instance and returns a pointer to it. Returns NULL
   * if something goes wrong.
   *
   * @param path [std::string] the absolute path to the dynamic library file
   * @param mode [UInt32] a bitmap of loading modes with platform-specific
   * meaning
   */
  static DynamicLibrary *load(const std::string &path, UInt32 mode,
                              std::string &errorString);

  /**
   * Loads a dynamic library file, stores the handle in a heap-allocated
   * DynamicLibrary instance and returns a pointer to it. Returns NULL
   * if something goes wrong.
   *
   * @param path [std::string] the absolute path to the dynamic library file
   * @param errorString [std::string] error message if load failed
   * @retval the DynamicLibrary pointer on success or NULL on failure
   */
  static DynamicLibrary *load(const std::string &path,
                              std::string &errorString);
  ~DynamicLibrary();

  /**
   * Gets a symbols from a loaded dynamic library.
   * Returns the symbol (usually a function pointer) as
   * a void *. The caller is responsible for casting to the
   * right type. Returns NULL if something goes wrong.
   *
   * @param name [std::string] the requested symbol name.
   */
  void *getSymbol(const std::string &name);

private:
  DynamicLibrary();

  DynamicLibrary(void *handle);
  DynamicLibrary(const DynamicLibrary &);

private:
  void *handle_;
};

} // namespace nupic

#endif // NTA_DYNAMIC_LIBRARY_HPP
