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
Environment Implementation
*/

#include <algorithm> // std::transform
#include <apr-1/apr_env.h>
#include <apr-1/apr_general.h>
#include <cctype> // toupper
#include <nupic/os/Env.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

bool Env::get(const std::string &name, std::string &value) {
  // @todo remove apr initialization when we have global initialization
  apr_status_t status = apr_initialize();
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::get -- Unable to initialize APR"
              << " name = " << name;
    return false;
  }

  // This is annoying. apr_env_get doesn't actually use the memory
  // pool it is given. But we have to set it up because the API
  // requires it and might use it in the future.
  apr_pool_t *poolP;
  status = apr_pool_create(&poolP, nullptr);
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::get -- Unable to create a pool"
              << " name = " << name;
    return false;
  }

  char *cvalue;
  bool returnvalue = false;
  status = apr_env_get(&cvalue, name.c_str(), poolP);
  if (status != APR_SUCCESS) {
    returnvalue = false;
  } else {
    returnvalue = true;
    value = cvalue;
  }
  apr_pool_destroy(poolP);
  return returnvalue;
}

void Env::set(const std::string &name, const std::string &value) {
  // @todo remove apr initialization when we have global initialization
  apr_status_t status = apr_initialize();
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::set -- Unable to initialize APR"
              << " name = " << name << " value = " << value;
    // ok to return. Haven't created a pool yet
    return;
  }

  apr_pool_t *poolP;
  status = apr_pool_create(&poolP, nullptr);
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::set -- Unable to create a pool."
              << " name = " << name << " value = " << value;
    // ok to return. Haven't created a pool yet.
    return;
  }

  status = apr_env_set(name.c_str(), value.c_str(), poolP);
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::set -- Unable to set variable " << name << " to "
              << value;
  }

  apr_pool_destroy(poolP);
  return;
}

void Env::unset(const std::string &name) {
  // @todo remove apr initialization when we have global initialization
  apr_status_t status = apr_initialize();
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::unset -- Unable to initialize APR."
              << " name = " << name;
    return;
  }

  apr_pool_t *poolP;
  status = apr_pool_create(&poolP, nullptr);
  if (status != APR_SUCCESS) {
    NTA_THROW << "Env::unset -- Unable to create a pool."
              << " name = " << name;
    return;
  }

  status = apr_env_delete(name.c_str(), poolP);
  if (status != APR_SUCCESS) {
    // not a fatal error because may not exist
    NTA_WARN << "Env::unset -- Unable to delete " << name;
  }
  apr_pool_destroy(poolP);
  return;
}

char **Env::environ_ = nullptr;

#if defined(NTA_OS_DARWIN)
#include <crt_externs.h>
#else
extern char **environ;
#endif

char **Env::getenv() {
  if (environ_ != nullptr)
    return environ_;

#if defined(NTA_OS_DARWIN)
  environ_ = *_NSGetEnviron();
#else
  environ_ = environ;
#endif

  return environ_;
}

static std::string
_getOptionEnvironmentVariable(const std::string &optionName) {
  std::string result = "NTA_";
  result += optionName;
  std::transform(result.begin(), result.end(), result.begin(), toupper);
  return result;
}

bool Env::isOptionSet(const std::string &optionName) {
  std::string envName = _getOptionEnvironmentVariable(optionName);
  std::string value;
  bool found = get(envName, value);
  return found;
}

std::string Env::getOption(const std::string &optionName,
                           std::string defaultValue) {
  std::string envName = _getOptionEnvironmentVariable(optionName);
  std::string value;
  bool found = get(envName, value);
  if (!found)
    return defaultValue;
  else
    return value;
}
