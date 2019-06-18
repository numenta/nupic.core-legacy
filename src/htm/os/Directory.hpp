/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2018, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file */

#ifndef NTA_DIRECTORY_HPP
#define NTA_DIRECTORY_HPP

//----------------------------------------------------------------------

#include <string>
#include <htm/os/Path.hpp>
#include <htm/os/ImportFilesystem.hpp>  // defines fs, er, etc.

//----------------------------------------------------------------------

namespace htm {
class Path;

class Directory {
public:
	// check if a directory exists
	static bool exists(const std::string &path);

	// true if the directory is empty
	static bool empty(const std::string &path);

	// return the amount of available space on this path's device.
	static Size free_space(const std::string & path);

	// get current working directory
	static std::string getCWD();

	// set current working directories
	static void setCWD(const std::string &path);  // be careful about using this.

	// Copy directory tree rooted in 'source' to 'destination'
	static void copyTree(const std::string &source, const std::string &destination);

	// Remove directory tree rooted in 'path'
	static bool removeTree(const std::string &path, bool noThrow = false);

	// Create directory 'path' including all parent directories if missing
	// returns the first directory that was actually created.
	//
	// For example if path is /A/B/C/D
	//    if /A/B/C/D exists it returns ""
	//    if /A/B exists it returns /A/B/C
	//    if /A doesn't exist it returns /A/B/C/D
	//
	// Failures will throw an exception
	static void create(const std::string &path, bool otherAccess = false, bool recursive = false);

	static std::string list(const std::string &path, std::string indent = "");
};


struct Entry {
  enum Type { FILE, DIRECTORY, LINK, OTHER };

  Type type;
  std::string path;         // full absolute path
  std::string filename;     // just the filename and extension or directory name
};

class Iterator {
public:
  Iterator(const htm::Path & path);
  Iterator(const std::string & path);
  ~Iterator() {}

  // Resets directory to start. Subsequent call to next()
  // will retrieve the first entry
  void reset();

  // get next directory entry
  Entry *next(Entry &e);

private:
  Iterator() = delete;
  Iterator(const Iterator &) = delete;

private:
  fs::path p_;
  fs::directory_iterator current_;
  fs::directory_iterator end_;

};
} // namespace htm

#endif // NTA_DIRECTORY_HPP


