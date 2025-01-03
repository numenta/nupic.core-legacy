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
 */

#include <algorithm>
#include <apr-1/apr_file_io.h>
#include <apr-1/apr_time.h>
#include <nupic/os/Directory.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>
#include <string>

#if defined(NTA_OS_WINDOWS)
#include <tchar.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nupic {
namespace Directory {
bool exists(const std::string &path) { return Path::exists(path); }

std::string getCWD() {
#if defined(NTA_OS_WINDOWS)
  wchar_t wcwd[APR_PATH_MAX];
  DWORD res = ::GetCurrentDirectoryW(APR_PATH_MAX, wcwd);
  NTA_CHECK(res > 0) << "Couldn't get current working directory. OS msg: "
                     << OS::getErrorMessage();
  std::string cwd = Path::unicodeToUtf8(std::wstring(wcwd));
  return cwd;
#else
  char cwd[APR_PATH_MAX];
  cwd[0] = '\0';
  char *res = ::getcwd(cwd, APR_PATH_MAX);
  NTA_CHECK(res != nullptr)
      << "Couldn't get current working directory. OS num: " << errno;
  return std::string(cwd);
#endif
}

bool empty(const std::string &path) {
  Entry dummy;
  return Iterator(path).next(dummy) == nullptr;
}

void setCWD(const std::string &path) {
  int res = 0;
#if defined(NTA_OS_WINDOWS)
  std::wstring wpath(Path::utf8ToUnicode(path));
  res = ::SetCurrentDirectoryW(wpath.c_str()) ? 0 : -1;
#else
  res = ::chdir(path.c_str());
#endif

  NTA_CHECK(res == 0) << "setCWD: " << OS::getErrorMessage();
}

static bool removeEmptyDir(const std::string &path, bool noThrow) {
  int res = 0;
#if defined(NTA_OS_WINDOWS)
  std::wstring wpath(Path::utf8ToUnicode(path));
  res = ::RemoveDirectoryW(wpath.c_str()) != FALSE ? 0 : -1;
#else
  res = ::rmdir(path.c_str());
#endif
  if (!noThrow) {
    NTA_CHECK(res == 0) << "removeEmptyDir: " << OS::getErrorMessage();
  }
  return (res == 0);
}

void copyTree(const std::string &source, const std::string &destination) {
  NTA_CHECK(Path::isDirectory(source));
  std::string baseSource(Path::getBasename(source));
  std::string dest(destination);
  dest = Path::join(dest, baseSource);
  if (!Path::exists(dest))
    Directory::create(dest, false, true);
  NTA_CHECK(Path::isDirectory(dest));

  Iterator i(source);
  Entry e;
  while (i.next(e)) {
    std::string fullSource(source);
    fullSource = Path::join(fullSource, e.path);
    Path::copy(fullSource, dest);
  }
}

bool removeTree(const std::string &path, bool noThrow) {
  bool success = true;
  NTA_CHECK(!path.empty()) << "Can't remove directory with no name";
  {
    // The scope is necessary to make sure the destructor
    // of the Iterator releases the directory so that
    // removeEmptyDir() will succeed.
    Iterator i(path);
    Entry e;
    while (i.next(e)) {
      Path fullPath = Path(path) + Path(e.path);
      if (e.type == Entry::DIRECTORY) {
        bool subResult = removeTree(std::string(fullPath), noThrow);
        success = success && subResult;
      } else {
        apr_status_t st = ::apr_file_remove(fullPath, nullptr);
        if (st != APR_SUCCESS) {
          if (noThrow)
            success = false;
          else {
            NTA_THROW << "Directory::removeTree() failed. "
                      << "Unable to remove the file'" << fullPath << "'. "
                      << "OS msg: " << OS::getErrorMessage();
          }
        }
      }
    }
  }

  bool subResult = removeEmptyDir(path, noThrow);
  success = success && subResult;
  // Check 3 times the directory is really gone
  // (needed for unreliable file systems)
  for (int i = 0; i < 3; ++i) {
    if (!Directory::exists(path))
      return success;
    // sleep for a second
    if (i < 2)
      ::apr_sleep(1000 * 1000);
  }
  if (!noThrow) {
    NTA_THROW << "Directory::removeTree() failed. "
              << "Unable to remove empty dir: "
              << "\"" << path << "\"";
  }
  return false;
}

// Create directory recursively (creates parent if doesn't exist)
// Helper function for create(.., recursive=true)
static std::string createRecursive(const std::string &path, bool otherAccess) {
  /// TODO: When the directory exists, confirm or update its permissions.

  NTA_CHECK(!path.empty()) << "Can't create directory with no name";
  std::string p = Path::makeAbsolute(path);

  if (Path::exists(p)) {
    if (!Path::isDirectory(p)) {
      NTA_THROW << "Directory::create -- path " << path
                << " already exists but is not a directory";
    }
    // Empty string return terminates the recursive call because "" has no
    // parent
    return "";
  }

  std::string result(p);
  std::string parent = Path::getParent(p);
  if (!Directory::exists(parent)) {
    result = createRecursive(parent, otherAccess);
  }

  create(p, otherAccess, false);
  return result;
}

void create(const std::string &path, bool otherAccess, bool recursive) {
  /// TODO: When the directory exists, confirm or update its permissions.

  if (recursive) {
    createRecursive(path, otherAccess);
    return;
  }

  // non-recursive case
  bool success = true;
#if defined(NTA_OS_WINDOWS)
  std::wstring wPath = Path::utf8ToUnicode(path);
  success = ::CreateDirectoryW(wPath.c_str(), NULL) != FALSE;
  if (!success) {
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
      // Not a hard error, due to potential race conditions.
      std::cerr << "Path '" << path
                << "' exists. "
                   "Possible race condition."
                << std::endl;
      success = Path::isDirectory(path);
    }
  }

#else
  int permissions = S_IRWXU;
  if (otherAccess) {
    permissions |= (S_IRWXG | S_IROTH | S_IXOTH);
  }
  int res = ::mkdir(path.c_str(), permissions);
  if (res != 0) {
    if (errno == EEXIST) {
      // Not a hard error, due to potential race conditions.
      std::cerr << "Path '" << path
                << "' exists. "
                   "Possible race condition."
                << std::endl;
      success = Path::isDirectory(path);
    } else {
      success = false;
    }
  } else
    success = true;
#endif

  if (!success) {
    NTA_THROW << "Directory::create -- failed to create directory \"" << path
              << "\".\n"
              << "OS msg: " << OS::getErrorMessage();
  }
}

Iterator::Iterator(const Path &path) { init(std::string(path)); }

Iterator::Iterator(const std::string &path) { init(path); }

void Iterator::init(const std::string &path) {
  apr_status_t res = ::apr_pool_create(&pool_, nullptr);
  NTA_CHECK(res == 0) << "Can't create pool";
  std::string absolutePath = Path::makeAbsolute(path);
  res = ::apr_dir_open(&handle_, absolutePath.c_str(), pool_);
  NTA_CHECK(res == 0) << "Can't open directory " << path
                      << ". OS num: " << APR_TO_OS_ERROR(res);
}

Iterator::~Iterator() {
  apr_status_t res = ::apr_dir_close(handle_);
  ::apr_pool_destroy(pool_);
  if (res != 0) {
    NTA_WARN << "Couldn't close directory."
             << " OS num: " << APR_TO_OS_ERROR(res);
  }
}

void Iterator::reset() {
  apr_status_t res = ::apr_dir_rewind(handle_);
  NTA_CHECK(res == 0) << "Couldn't reset directory iterator."
                      << " OS num: " << APR_TO_OS_ERROR(res);
}

Entry *Iterator::next(Entry &e) {
  apr_int32_t wanted = APR_FINFO_LINK | APR_FINFO_NAME | APR_FINFO_TYPE;
  apr_status_t res = ::apr_dir_read(&e, wanted, handle_);

  // No more entries
  if (APR_STATUS_IS_ENOENT(res))
    return nullptr;

  if (res != 0) {
    NTA_CHECK(res == APR_INCOMPLETE) << "Couldn't read next dir entry."
                                     << " OS num: " << APR_TO_OS_ERROR(res);
    NTA_CHECK(((e.valid & wanted) | APR_FINFO_LINK) == wanted)
        << "Couldn't retrieve all fields. Valid mask=" << e.valid;
  }

  e.type = (e.filetype == APR_DIR) ? Directory::Entry::DIRECTORY
                                   : Directory::Entry::FILE;
  e.path = e.name;

  // Skip '.' and '..' directories
  if (e.type == Directory::Entry::DIRECTORY &&
      (e.name == std::string(".") || e.name == std::string("..")))
    return next(e);
  else
    return &e;
}
} // namespace Directory
} // namespace nupic
