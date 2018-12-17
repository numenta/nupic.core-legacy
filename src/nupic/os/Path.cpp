/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

/** @file
 */

#include <nupic/os/Directory.hpp> // also includes <filesystem> or <experimental/filesystem>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/StringUtils.hpp>  // for trim
#include <algorithm> // replace()
#include <sstream>
#include <string>
#include <iterator>
#include  <stdio.h>
#include  <stdlib.h>
#include <fstream>
#if defined(NTA_OS_WINDOWS)
  #include  <io.h>
#else
  #include <sys/stat.h>
  #include <unistd.h>
#endif


namespace nupic {

const char *Path::parDir = "..";
#if defined(NTA_OS_WINDOWS)
const char *Path::sep = "\\";
const char *Path::pathSep = ";";
#else
const char *Path::sep = "/";
const char *Path::pathSep = ":";
#endif

bool Path::exists(const std::string &path) { return fs::exists(path); }

bool Path::equals(const std::string &path1, const std::string &path2) {
  std::string s1 = normalize(path1);
  std::string s2 = normalize(path2);
  return (s1 == s2);
}

bool Path::isFile(const std::string &path) { return fs::is_regular_file(path); }

bool Path::isDirectory(const std::string &path) {
  return fs::is_directory(path);
}

bool Path::isSymbolicLink(const std::string &path) {
  return fs::is_symlink(path);
}

bool Path::isAbsolute(const std::string &path) {
#if defined(NTA_OS_WINDOWS)
  // TODO: it is unclear what is_absolute() returns in Windows so do our own.
  if (path.length() == 2 && isalpha(path[0]) && path[1] == ':')
    return true; //  c:
  if (path.length() >= 3 && path[0] == '\\' && path[1] == '\\' &&
      isalpha(path[3]))
    return true; // \\net
#endif
  return fs::path(path).is_absolute();
}

bool Path::areEquivalent(const std::string &path1, const std::string &path2) {
  fs::path p1(path1);
  fs::path p2(path2);
  if (!fs::exists(p1) || !fs::exists(p2))
    return false;
  // must resolve to the same existing filesystem and file entry to match.
  return fs::equivalent(p1, p2);
}

std::string Path::getParent(std::string path) {
  //std::cerr << "Initial Path " << path << std::endl;
  path = normalize(path);
  //std::cerr << "Normalized Path " << path << std::endl;
  if (path.empty() || path == ".") return "..";
  if (path.length() >= 2 && path.substr(path.length() - 2) == "..") {
    return (path + std::string(Path::sep) + "..");
  }
  fs::path p(path);
  if (p.root_path() == p) return p.string();
  if (p.string().back() == Path::sep[0])
    p = p.parent_path(); // remove trailing sep if not root
  p = p.parent_path();
  //std::cerr << "Parent Path " << p << std::endl;
  if (p.empty())
    return ".";
  return p.string();
}

std::string Path::getExtension(const std::string &path) {
  if (path.empty())
    return path;
  fs::path p(path);
  if (p.has_extension())
    return p.extension().string().substr(1); // do not include the .
  return "";
}

Size Path::getFileSize(const std::string &path) {
  return (Size)fs::file_size(path);
}


/**
 *  A path can be normalized by following this algorithm:  (from C++17 std::filesystem::path)
 *      1) If the path is empty, stop (normal form of an empty path is an empty path)
 *      2) Replace each directory-separator (which may consist of multiple slashes) with a single path::preferred_separator.
 *      3) Replace each slash character in the root-name with path::preferred_separator.
 *      4) Remove each dot and any immediately following directory-separator.
 *      5) Remove each non-dot-dot filename immediately followed by a directory-separator and a dot-dot, along with any immediately following directory-separator.
 *      6) If there is root-directory, remove all dot-dots and any directory-separators immediately following them.
 *      7) If the last filename is dot-dot, remove any trailing directory-separator.
 *      8) If the path is empty, add a dot (normal form of ./ is .)
 *
 * TODO: I expected to use fs::system_complete() or fs::canonical()
 *       for this but it requires the path to exist.
 *       The function path.lexically_normal() does the job perfectly
 *       but it is not available in boost or <experimental/filesystem>
 *       So for now we roll our own version.
 *       Replace this when everyone is using C++17 or better.
 */
std::string Path::normalize(const std::string &path) {
  std::string trimmed_path = StringUtils::trim(path);
  if (trimmed_path.empty()) return ".";
  std::replace(trimmed_path.begin(), trimmed_path.end(), '\\', '/'); // in-place replace
  fs::path p(trimmed_path);
  //  p = p.lexically_normal();
  //  if (p.string().back() == Path::sep[0])
  //    p = p.parent_path(); // remove trailing sep if not root
  //  return p.string();

  std::vector<std::string> normal_p;
  // iterate the path and build a list of path elements.
  fs::path::iterator iter = p.begin();
  if (p.has_root_name()) normal_p.push_back((iter++)->string());
  if (p.has_root_path()) normal_p.push_back((iter++)->string());
  size_t j = normal_p.size();  // minimum size.
  for ( ; iter != p.end(); iter++) {
  	std::string ele = StringUtils::trim(iter->string());
    if (ele == "." || ele == "") continue;
    if (ele == ".." && normal_p.size() > j && normal_p.back() != "..") {
      normal_p.pop_back();
      continue;
	}
	normal_p.push_back(ele);
  }
  fs::path new_p;
  for(auto& ele : normal_p) {
  	new_p /= ele;
  }
  new_p = new_p.make_preferred();
  std::string result = new_p.string();
  return result;

}


std::string Path::makeAbsolute(const std::string &path) {
  return fs::absolute(path).string();
}


// unicodeToUtf8() and utf8ToUnicode() moved to StringUtil.cpp

Path::StringVec Path::split(const std::string &path) {
  /**
   * Don't use boost::tokenizer because we need to handle the prefix specially.
   * Handling the prefix is messy on windows, but this is the only place we have
   * to do it
   * TODO: replace this with filesystem iterator if it does the right thing.
   */
  StringVec parts;
  std::string::size_type curpos = 0;
  if (path.size() == 0)
    return parts;

#if defined(NTA_OS_WINDOWS)
  // prefix may be 1) "\", 2) "\\", 3) "[a-z]:", 4) "[a-z]:\"
  if (path.size() == 1) {
    // captures both "\" and "a"
    parts.push_back(path);
    return parts;
  }
  if (path[0] == '\\') {
    if (path[1] == '\\') {
      // case 2
      parts.push_back("\\\\");
      curpos = 2;
    } else {
      // case 1
      parts.push_back("\\");
      curpos = 1;
    }
  } else {
    if (path[1] == ':') {
      if (path.size() > 2 && path[2] == '\\') {
        // case 4
        parts.push_back(path.substr(0, 3));
        curpos = 3;
      } else {
        parts.push_back(path.substr(0, 2));
        curpos = 2;
      }
    }
  }
#else
  // only possible prefix is "/"
  if (path[0] == '/') {
    parts.push_back("/");
    curpos++;
  }
#endif

  // simple tokenization based on separator. Note that "foo//bar" -> "foo", "",
  // "bar"
  std::string::size_type newpos;
  while (curpos < path.size() && curpos != std::string::npos) {
    // Be able to split on either separator including mixed separators on
    // Windows
#if defined(NTA_OS_WINDOWS)
    std::string::size_type p1 = path.find("\\", curpos);
    std::string::size_type p2 = path.find("/", curpos);
    newpos = p1 < p2 ? p1 : p2;
#else
    newpos = path.find(Path::sep, curpos);
#endif

    if (newpos == std::string::npos) {
      parts.push_back(path.substr(curpos));
      curpos = newpos;
    } else {
      // note: if we have a "//" then newpos == curpos and this string is empty
      if (newpos != curpos) {
        parts.push_back(path.substr(curpos, newpos - curpos));
      }
      curpos = newpos + 1;
    }
  }

  return parts;
}

std::string Path::getBasename(const std::string &path) {
  std::string name = "";
  if (!path.empty()) {
    char last_ch = path[path.length() - 1];
    if (last_ch == '/' || last_ch == '\\')
      name = ".";
    else {
      fs::path p(path);
      if (p.has_filename())
        name = p.filename().string();
    }
  }
  //std::cerr << "Basename of " << path << " is " << name << std::endl;

  return name;
}


// determine if it is something like  C:\ on windows or / or linux. or perhaps //hostname
bool Path::isRootdir(const std::string &s) {
  fs::path p(s);
  return (p.root_path().string() == s);
}


std::string Path::join(StringVec::const_iterator begin,
                       StringVec::const_iterator end) {
  if (begin == end)
    return "";

  if (begin + 1 == end)
    return std::string(*begin);

  std::string path(*begin);
#if defined(NTA_OS_WINDOWS)
  if (path[path.length() - 1] != Path::sep[0])
    path += Path::sep;
#else
  // Treat first element specially (on Unix)
  // it may be a prefix, which is not followed by "/"
  if (!Path::isPrefix(*begin))
    path += Path::sep;
#endif
  begin++;

  while (begin != end) {
    path += *begin;
    begin++;
    if (begin != end) {
      path += Path::sep;
    }
  }

  return path;
}

std::string Path::join(const std::string &path1, const std::string &path2) {
  return path1 + Path::sep + path2;
}

std::string Path::join(const std::string &path1, const std::string &path2,
                       const std::string &path3) {
  return path1 + Path::sep + path2 + Path::sep + path3;
}

std::string Path::join(const std::string &path1, const std::string &path2,
                       const std::string &path3, const std::string &path4) {
  return path1 + Path::sep + path2 + Path::sep + path3 + Path::sep + path4;
}


void Path::copy(const std::string &source, const std::string &destination) {
  NTA_CHECK(!source.empty()) << "Can't copy from an empty source";

  NTA_CHECK(!destination.empty()) << "Can't copy to an empty destination";

  NTA_CHECK(!areEquivalent(source, destination))
      << "Source and destination must be different";

  if (isDirectory(source)) {
    Directory::copyTree(source, destination);
    return;
  }
  fs::path parent = fs::path(destination).parent_path();
  er::error_code ec;
  if (!parent.string().empty() && !fs::exists(parent)) {
    fs::create_directories(parent, ec);
    NTA_CHECK(!ec) << "Path::copy - failure creating destination path '"
                   << destination << "'" << ec.message();
  }
  fs::copy_file(source, destination, ec);
  NTA_CHECK(!ec) << "Path::copy - failure copying file '" << source << "' to '"
                 << destination << "'" << ec.message();
}

void Path::setPermissions(const std::string &path, bool userRead,
                          bool userWrite, bool groupRead, bool groupWrite,
                          bool otherRead, bool otherWrite) {

  if (Path::isDirectory(path)) {
    fs::perms prms =
	    (userRead ? fs::perms::owner_exe | fs::perms::owner_read : FS_PermNone) |
        (userWrite ? fs::perms::owner_all : FS_PermNone) |
        (groupRead ? FS_OthersExec | fs::perms::group_read : FS_PermNone) |
        (groupWrite ? fs::perms::group_all : FS_PermNone) |
        (otherRead ? FS_OthersExec | fs::perms::others_read  : FS_PermNone) |
        (otherWrite ? fs::perms::others_all : FS_PermNone);
    fs::permissions(path, prms);

    Directory::Iterator iter(path);
    Directory::Entry e;
    while (iter.next(e)) {
      setPermissions(e.path,
	                 userRead,
					 userWrite,
					 groupRead,
					 groupWrite,
                     otherRead,
					 otherWrite);
    }
  } else {
    fs::perms prms =
	    (userRead ? fs::perms::owner_read : FS_PermNone) |
        (userWrite ? fs::perms::owner_write : FS_PermNone) |
        (groupRead ? fs::perms::group_read : FS_PermNone) |
        (groupWrite ? fs::perms::group_write : FS_PermNone) |
        (otherRead ? fs::perms::others_read : FS_PermNone) |
        (otherWrite ? fs::perms::others_write : FS_PermNone);
    fs::permissions(path, prms);
  }
}

void Path::remove(const std::string &path) {
  NTA_CHECK(!path.empty()) << "Can't remove an empty path";

  // Just return if it doesn't exist already
  if (!Path::exists(path))
    return;

  if (isDirectory(path)) {
    Directory::removeTree(path);
    return;
  }
  er::error_code ec;
  fs::remove(path, ec);
  NTA_CHECK(!ec) << "Path::remove - failure removing file '" << path << "'"
                 << ec.message();
}

void Path::rename(const std::string &oldPath, const std::string &newPath) {
  NTA_CHECK(!oldPath.empty() && !newPath.empty())   << "Can't rename to/from empty path";
  fs::path oldp = fs::absolute(oldPath);
  fs::path newp = fs::absolute(newPath);

  er::error_code ec;
  fs::rename(oldp, newp, ec);
  NTA_CHECK(!ec) << "Path::remove - failure renaming file '" << oldp.string()
                 << "' to '" << newp.string() << "'" << ec.message();
}

void Path::write_all(const std::string &filename, const std::string &value) {
  std::ofstream f;
  f.open(filename.c_str());

  f << value;
  f.close();
}

std::string Path::read_all(const std::string &filename) {
  std::string s;
  std::ifstream f;
  f.open(filename.c_str());
  f >> s;
  return s;
}





// Global operators
Path operator/(const Path & p1, const Path & p2) { return Path(std::string(*p1) + Path::sep + std::string(*p2)); }
Path operator/(const std::string & p1, const Path & p2) { return Path(p1 + Path::sep + std::string(*p2)); }
Path operator/(const Path & p1, const std::string & p2) { return Path(std::string(*p1) + Path::sep + p2); }
Path operator+(const Path & p1, const Path & p2) { return Path(std::string(*p1) + std::string(*p2)); }
Path operator+(const std::string & p1, const Path & p2) { return Path(p1 + std::string(*p2)); }
Path operator+(const Path & p1, const std::string & p2) { return Path(std::string(*p1) + p2); }


/*******************************************
* This function returns the full path of the executable that is running this code.
* see https://stackoverflow.com/questions/1023306/finding-current-executables-path-without-proc-self-exe
*     https://stackoverflow.com/questions/1528298/get-path-of-executable
* The above links show that this is difficult to do and not 100% reliable in the general case.
* However, the boost solution is one of the best.
* /
std::string Path::getExecutablePath()
{
  er::error_code ec;
  er::path p = boost::dll::program_location(ec);
  NTA_CHECK(!ec) << "Path::getExecutablePath() Fail. " << ec.message();
  return p.string();
}
****** Removed until we find we need this function, dek 06/2018 ********/
} // namespace nupic
