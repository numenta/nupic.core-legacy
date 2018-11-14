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

/** @file
 * Implementation for Directory test
 */

#include <gtest/gtest.h>
#include <nupic/os/Directory.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>

#define VERBOSE std::cerr << "[          ] "


#if defined(NTA_OS_WINDOWS)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#include <algorithm> // sort


#ifndef PATH_MAX
#define PATH_MAX        4096    /* # chars in a path name including nul */
#endif


using namespace std;
using namespace nupic;
namespace testing {

static std::string getCurrDir() {
    char buff[PATH_MAX];
#if defined(NTA_OS_WINDOWS)
    DWORD res = ::GetCurrentDirectoryA(PATH_MAX - 1, (LPSTR)buff);
    NTA_CHECK(res > 0) << OS::getErrorMessage();

#else
    char * s = ::getcwd(buff, PATH_MAX - 1);
    NTA_CHECK(s != nullptr) << OS::getErrorMessage();
#endif
    return buff;
}

std::string sep(Path::sep);

TEST(DirectoryTest, Existence) {

  ASSERT_TRUE(!Directory::exists("No such dir"));
  if (Directory::exists("dir_0"))
    Directory::removeTree("dir_0");
  Directory::create("dir_0");
  ASSERT_TRUE(Directory::exists("dir_0"));

  Directory::removeTree("dir_0");
}


TEST(DirectoryTest, setCWD) {
  std::string baseDir = Path::makeAbsolute(getCurrDir());

  // make sure the TestOutputDir is entirely empty.
  if (!Path::exists("TestOutputDir"))
    Directory::create("TestOutputDir");
  ASSERT_TRUE(Path::exists("TestOutputDir"));

  Directory::setCWD("TestOutputDir");
  std::string cwd1 = Path::makeAbsolute(getCurrDir());

  std::string cwd2 = baseDir + Path::sep + std::string("TestOutputDir");
  ASSERT_STREQ(cwd1.c_str(), cwd2.c_str());

  // restore CWD
  Directory::setCWD(baseDir);

  ASSERT_EQ(baseDir, getCurrDir());
  Directory::removeTree("TestOutputDir");
  ASSERT_FALSE(Path::exists("TestOutputDir"));
}


TEST(DirectoryTest, getCWD) {
  ASSERT_EQ(getCurrDir(), Directory::getCWD());
}


TEST(DirectoryTest, RemoveTreeAndCreate) {

  std::string p = Path::makeAbsolute("TestOutputDir");
  std::string d = p + sep + "someSubDir";
  if (Path::exists(p))
    Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));
  ASSERT_THROW(Directory::create(d), exception); // nonrecursive create should fail
  Directory::create(d, false, true); // recursive
  ASSERT_TRUE(Path::exists(d));
  // Note: This will fail if you have your Windows Explorer open inside that folder
  //       or the 'someDir' line selected.
  Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(d));
  ASSERT_TRUE(!Path::exists(p));
}

TEST(DirectoryTest, CopyTree) {
  std::string p = Path::makeAbsolute("TestOutputDir");
  std::string a = Path::join(p, "A");
  std::string b = Path::join(p, "B");

  if (Path::exists(p))
    Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));

  // Create Directory:  TestOutputDir/A
  Directory::create(a, false, true /* recursive */);
  ASSERT_TRUE(Path::exists(a));

  // Create Directory:   TestOutput/B
  Directory::create(b);
  ASSERT_TRUE(Path::exists(b));

  // Create a file: TestOutput/B/1.txt
  std::string src(Path::join(b, "1.txt"));
  if (Path::exists(src))
    Path::remove(src);
  ASSERT_TRUE(!Path::exists(src));
  Path::write_all(src, "12345");
  ASSERT_TRUE(Path::exists(src));

  // copy directory TestOutputDir/B into TestOuputDir/A
  Directory::copyTree(b, a);

  std::string ls = Directory::list("TestOutputDir");
  std::cerr << "directory tree \n" << ls << "\n";
  // the file should exist in both directories
  //   TestOutputDir
  //       A
  //          1.txt
  //       B
  //          1.txt
  //
  ASSERT_TRUE(Path::exists(Path::join(a, "1.txt")));
  ASSERT_TRUE(Path::exists(Path::join(b, "1.txt")));

  std::string dest  = a + sep + "1.txt";
  ASSERT_STREQ(Path::read_all(dest).c_str(), "12345") << "Content of dest file not as expected.";

  // clean up
  Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));
}

TEST(DirectoryTest, Iterator) {
  if (Directory::exists("TestOutputDir/A"))
    Directory::removeTree("TestOutputDir/A");
  Directory::create("TestOutputDir/A/B", false, true);
  Directory::create("TestOutputDir/A/C", false, true);

  Directory::Iterator di("TestOutputDir/A");
  Directory::Entry entry;
  Directory::Entry * e = nullptr;

  vector<std::string> subdirs;
  e = di.next(entry);
  ASSERT_TRUE(e != nullptr);
  ASSERT_TRUE(e->type == Directory::Entry::DIRECTORY);
  subdirs.push_back(e->filename);
  string first = e->filename;

  e = di.next(entry);
  ASSERT_TRUE(e != nullptr);
  ASSERT_TRUE(e->type == Directory::Entry::DIRECTORY);
  subdirs.push_back(e->filename);


  e = di.next(entry);
  ASSERT_TRUE(e == nullptr);

  // Get around different directory iteration orders on different platforms
  std::sort(subdirs.begin(), subdirs.end());
  ASSERT_TRUE(subdirs[0] == "B");
  ASSERT_TRUE(subdirs[1] == "C");

  // check that after reset first entry is returned again
  di.reset();
  e = di.next(entry);
  ASSERT_TRUE(e != nullptr);
  ASSERT_TRUE(e->type == Directory::Entry::DIRECTORY);
  ASSERT_TRUE(e->filename == first);

  // Cleanup test dirs
  ASSERT_TRUE(Path::exists("TestOutputDir/A"));
  Directory::removeTree("TestOutputDir");
  ASSERT_TRUE(!Path::exists("TestOutputDir"));
}

} // namespace

