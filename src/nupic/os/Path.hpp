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

#ifndef NTA_PATH_HPP
#define NTA_PATH_HPP

//----------------------------------------------------------------------

#include <nupic/types/Types.hpp>
#include <string>
#include <vector>

//----------------------------------------------------------------------

namespace nupic {

/**
 * @b Responsibility:
 *  1. Represent a cross-platform path to a filesystem object
 *     (file, directory, symlink)
 *
 *  2. Provide a slew of of path manipulation operations
 *
 * @b Rationale:
 *  File system paths are used a lot. It makes sense to have
 *  a cross-platform class with a nice interface tailored to our needs.
 *  In particular operations throw NTA::Exception on failure and
 *  don't return error codes, which is alligned nicely with the
 *  way we handle errors.
 *
 *  Operations are both static and instance methods (use single implementation).
 *
 * @b Resource/Ownerships:
 *  1. A path string for the instance.
 *
 * @b Notes:
 *  The Path() constructors don't try to validate the path string
 *  for efficiency reasons (it's complicated too). If you pass
 *  an invalid path string it will fail when you actually try to use
 *  the resulting path.
 *
 *  The error handling strategy is to return error NULLs and not to throw
 * exceptions. The reason is that it is a very generic low-level class that
 * should not be aware and depend on the runtime's error handling policy. It may
 * be used in different contexts like tools and utilities that may utilize a
 * different error handling strategy. It is also a common idiom to return NULL
 * from a failed factory method.
 *
 * @b Performance:
 *  The emphasis is on code readability and ease of use. Performance takes
 * second place, because the critical path of our codebase doesn't involve a lot
 * of path manipulation. In particular, simple ANSI C or POSIX cross-platform
 * implementation is often preffered to calling specific platform APIs.
 *
 *  Note, that constructing a Path object (or calling the Path instance methods)
 * involve an extra copy of the path string into the new Path instance. Again,
 * this is not prohibitive in most cases. If you are concerned use plain strings
 * and the static methods.
 *
 */
class Path {
public:

  static const char *sep;
  static const char *pathSep;
  static const char *parDir;
  typedef std::vector<std::string> StringVec;

  /**
  * This will first convert the path to absolute, then return the full parent path.
  */
  static std::string getParent(std::string path);

  /**
   * getBasename(foo/bar.baz) -> bar.baz
   */
  static std::string getBasename(const std::string &path);

  /**
   * getExtension(foo/bar.baz) -> .baz
   */
  static std::string getExtension(const std::string &path);

  /**
  * Removes dot and dot-dot symbolic links from path
  * normalize("/foo/bar/..") -> "/foo"
  * normalize("/foo/../bar") -> "/bar"
  * normalize("..") -> parent of current path
  * normalize(".") -> current path
  * Note: may convert from relative to absolute if ".." backs up over current directory
  **/
  static std::string normalize(const std::string &path);

  /**
   * makeAbsolute(path) ->
   * if isAbsolute(path) -> path
   * otherwise it combines cwd and path into an absolute path.
   */
  static std::string makeAbsolute(const std::string &path);


  /**
   * When splitting a path into components, the "prefix" has to be
   * treated specially. We do not store it in a separate data
   * structure -- the prefix is just the first element of the split.
   * No normalization is performed. We always have path == join(split(path))
   * except when there are empty components, e.g. foo//bar. Empty components
   * are omitted.
   * See the java.io.file module documentation for some good background
   * split("foo/bar/../quux") -> ("foo", "bar", "..", "quux")
   * split("/foo/bar/quux") -> ("/", "foo", "bar", "quux")
   * split("a:\foo\bar") -> ("a:\", "foo", "bar")
   * split("\\host\drive\file") -> ("\\", "host", "drive", "file")
   * split("/foo//bar/") -> ("/", "foo", "bar")
   * Note: this behavior is different from the original behavior.
   */
  static StringVec split(const std::string &path);

  /**
   * Construct a path from components. path == join(split(path))
   */
  static std::string join(StringVec::const_iterator begin,
                          StringVec::const_iterator end);
  /**
   * varargs through overloading
   */
  static std::string join(const std::string &path1, const std::string &path2);
  static std::string join(const std::string &path1, const std::string &path2,
                          const std::string &path3);
  static std::string join(const std::string &path1, const std::string &path2,
                          const std::string &path3, const std::string &path4);

  /**
  * path == "/" on unix
  * path == "\\" or path == "C:\\" on windows
  */
  static bool isRootdir(const std::string &path);
  static bool isPrefix(const std::string &path) { return isRootdir(path);} // an alias

  /**
   * isAbsolute("/foo/bar") -> true isAbsolute("foo")->false on Unix
   * is Absolute("a:\foo\bar") -> true isAbsolute("\foo\bar") -> false on windows
   */
  static bool isAbsolute(const std::string &path);


  /**
   * true if path exists. false is for broken links
   */
  static bool exists(const std::string &path);

  /**
  * true if both paths are lexically equal.
  * (unix formatted paths should match with Windows formatted paths).
  */
  static bool equals(const std::string& path1, const std::string& path2);

  /**
   * getFileSize throws an exception if does not exist or is a directory
   */
  static Size getFileSize(const std::string &path);

  /**
   * If source is a file, copy the file to the destination.
   * if a folder, copy entire folder recursivly.
   */
  static void copy(const std::string &source, const std::string &destination);

  /**
  * If source is a file, delete the file. if a directory, delete entire directory and all contents recursivly.
  */
  static void remove(const std::string &path);

  /**
   * move or change the file or directory name.
   */
  static void rename(const std::string &oldPath, const std::string &newPath);

  static bool isDirectory(const std::string &path);
  static bool isFile(const std::string &path);
  static bool isSymbolicLink(const std::string &path);

  // equivalent if both point to same file system and object, regardless of symbolic links.
  static bool areEquivalent(const std::string &path1, const std::string &path2);

  // Get a path to the currently running executable
  // don't think we need this...
  //static std::string getExecutablePath();

  /**
  * Set permissions on files and directories.
  * On Windows: All permissions except write are currently ignored. There is only a single write permission;
  *             setting write permission for owner, group, or others sets write permission for all, and
  *             removing write permission for owner, group, or others removes write permission for all.
  */
  static void setPermissions(const std::string &path,
                             bool userRead,
                             bool userWrite,
							 bool groupRead,
							 bool groupWrite,
                             bool otherRead,
							 bool otherWrite);




  /**
   * Read and Write entire contents of file.
   */
  static void write_all(const std::string& filename, const std::string& value);
  static std::string read_all(const std::string& filename);


  ////////////////////////////////////////////////////////////////////////////////

  Path(std::string path)       {  path_ = path; }
  const char* operator*() const { return path_.c_str(); }
  Path & operator +=(const Path & path) { path_ += Path::sep + path.path_; return *this; }
  bool operator==(const Path & other) { return equals(path_, other.path_); }  // lexical compare
  Path getParent() const       { return Path(getParent(path_)); }
  Path getBasename() const     { return Path(getBasename(path_)); }  // This is the filename with extension
  Path getExtension() const    { return Path(getExtension(path_)); }
  Path makeAbsolute() const    { return Path(makeAbsolute(path_)); }
  Path normalize() const       { return Path(normalize(path_)); }
  Size getFileSize() const     { return getFileSize(path_); }
  std::string string()const    { return path_; }
  const char* c_str() const    { return path_.c_str(); }

  void remove() const                                { remove(path_); }
  void copy(const std::string & destination) const   { copy(path_, destination); }
  void rename(const std::string & newPath)  { rename(path_, newPath); path_ = makeAbsolute(newPath); }
  void write_all(const std::string& value) { write_all(path_, value); }
  std::string read_all() { return read_all(path_); }

  bool isDirectory() const      { return isDirectory(path_); }
  bool isFile() const           { return isFile(path_); }
  bool isRootdir() const        { return isRootdir(path_); }
  bool isAbsolute() const       { return isAbsolute(path_); }
  bool isSymbolicLink() const   { return isSymbolicLink(path_); }
  bool isEmpty() const          { return path_.empty(); }
  bool exists() const           { return exists(path_); }

private:
  Path() = delete;	// disallow empty constructor.

private:
  std::string path_;
};

// Global operators
// (concatination of filename with separator)
Path operator/(const Path & p1, const Path & p2);
Path operator/(const std::string & p1, const Path & p2);
Path operator/(const Path & p1, const std::string & p2);
Path operator+(const Path & p1, const Path & p2);
Path operator+(const std::string & p1, const Path & p2);
Path operator+(const Path & p1, const std::string & p2);

} // namespace nupic

#endif // NTA_PATH_HPP


