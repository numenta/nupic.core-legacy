/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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

/** @file
 */

#include <algorithm>
#include <thread>

#include <htm/os/Directory.hpp>  // defines namespace fs
#include <htm/os/Path.hpp>
#include <htm/utils/Log.hpp>
#include <string>



namespace htm {
bool Directory::exists(const std::string &fpath) { return fs::exists(fpath); }

std::string Directory::getCWD() { return fs::current_path().string(); }

bool Directory::empty(const std::string &path) { return fs::is_empty(path); }
Size Directory::free_space(const std::string &path) {
  fs::space_info si = fs::space(path);
  return (Size)si.available; // disk space available to non-privalaged pocesses.
}

void Directory::setCWD(const std::string &path) { fs::current_path(path); }


// copy a directory recursively.
// It does not copy links.  It tries to retain permissions.
// If a file already exists in the destination it is overwritten.
// If the destination is an existing directory, the contents of
// the source directory are copied into the destination directory.
// Note: this is NOT like a cp command.
void Directory::copyTree(const std::string &source, const std::string &destination) {
  er::error_code ec;

  NTA_CHECK(Path::isDirectory(source)) << "copyTree() source is not a directory. " << source;
  if (Path::exists(destination)) {
    NTA_CHECK(Path::isDirectory(destination))
        << "copyTree() destination exists '" << destination
        << "' and it is not a directory.";
  } else {
#ifdef USE_BOOST_FILESYSTEM
    fs::path p(destination);
    fs::create_directory(p, ec);
#else
    // creates directory, copying permissions
    fs::create_directory(destination, source, ec);
#endif
    NTA_CHECK(!ec) << "copyTree: Could not create destination directory. '"
                   << destination << "' " << ec.message();
  }
  Iterator it(source);
  Entry entry;
  while (it.next(entry) != nullptr) {
    // Note: this does not copy links.
    std::string to = destination + Path::sep + entry.filename;
    if (entry.type == Entry::FILE) {
      fs::copy_file(entry.path, to, FS_Overwrite, ec);
    } else if (entry.type == Entry::DIRECTORY) {
      copyTree(entry.path, to);
    }
  }
}

bool Directory::removeTree(const std::string &path, bool noThrow) {
  er::error_code ec;
  if (fs::exists(path)) {
    if (fs::is_directory(path, ec)) {
      fs::remove_all(path, ec);
    }
    if (!noThrow) {
      NTA_CHECK(!ec) << "removeTree: " << ec.message();
    }
  }
  return (!ec);
}

// create directory
void Directory::create(const std::string &path, bool otherAccess, bool recursive) {
  NTA_CHECK(!path.empty())  << "Directory::create -- Can't create directory with no name";
  fs::path p = fs::absolute(path);
  if (fs::exists(p)) {
    if (!fs::is_directory(p)) {
      NTA_THROW << "Directory::create -- path " << path << " already exists but is not a directory";
    }
    return;
  } else {
    if (recursive) {
      er::error_code ec;
      if (!fs::create_directories(p, ec)) {
        NTA_CHECK(!ec) << "Directory::createRecursive: " << ec.message();
      }
    } else {
      // non-recursive case
      NTA_CHECK(fs::exists(p.parent_path())) << "Directory::create -- path " << path << " Parent directory does not exist.";
      er::error_code ec;
      fs::create_directory(p, ec);
      NTA_CHECK(!ec) << "Directory::create " << ec.message();
    }
  }
  // Set permissions on directory.
#if !defined(NTA_OS_WINDOWS)
  fs::perms prms(fs::perms::owner_all
    | (otherAccess ? (fs::perms::group_all | fs::perms::others_read | FS_OthersExec) : FS_PermNone));
  fs::permissions(p, prms);
#endif
}

std::string Directory::list(const std::string &path, std::string indent) {
  if (!fs::exists(path))
    return "";
  if (!Path::isDirectory(path))
    return indent + Path::getBasename(path) + "\n";
  std::string ls = indent +  Path::getBasename(path) + "\n";
  indent += "  ";
  Iterator di(path);
  Entry entry;
  Entry * e = nullptr;
  while ((e = di.next(entry))) {
    if (e->type == Entry::DIRECTORY)
      ls += Directory::list(e->path, indent);
    else
      ls += indent + e->filename + "\n";
  }
  return ls;
}

// Iterator functions
Iterator::Iterator(const htm::Path &path) {
  std::string pstr = path.c_str();
  p_ = fs::absolute(pstr);
  current_ = fs::directory_iterator(p_);
}
Iterator::Iterator(const std::string &path) {
  p_ = fs::absolute(path);
  current_ = fs::directory_iterator(p_);
}
void Iterator::reset() { current_ = fs::directory_iterator(p_); }
Entry * Iterator::next(Entry & e) {
  er::error_code ec;
  if (current_ == end_)  return nullptr;
  e.type = (fs::is_directory(current_->path(), ec)) ? Entry::DIRECTORY :
           (fs::is_regular_file(current_->path(), ec)) ? Entry::FILE :
           (fs::is_symlink(current_->path(), ec))? Entry::LINK : Entry::OTHER;
  e.path = current_->path().string();
  e.filename = current_->path().filename().string();
  current_++;
  return &e;
}



} // namespace htm
