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


#include <nupic/os/Directory.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/FStream.hpp>
#include <apr-1/apr.h>
#include <gtest/gtest.h>
#include <nupic/utils/Log.hpp>


#if defined(NTA_OS_WINDOWS)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#include <algorithm> // sort
using namespace std;
using namespace nupic;

static std::string getCurrDir()
{
    char buff[APR_PATH_MAX+1];
#if defined(NTA_OS_WINDOWS)
  DWORD res = ::GetCurrentDirectoryA(APR_PATH_MAX, (LPSTR)buff);
  NTA_CHECK(res > 0) << OS::getErrorMessage();

#else
  char * s = ::getcwd(buff, APR_PATH_MAX);
  NTA_CHECK(s != nullptr) << OS::getErrorMessage();
#endif
  return buff;  
}


std::string sep(Path::sep);

TEST(DirectoryTest, Existence)
{
  
  ASSERT_TRUE(!Directory::exists("No such dir"));
  if (Directory::exists("dir_0"))
    Directory::removeTree("dir_0");
  Directory::create("dir_0");
  ASSERT_TRUE(Directory::exists("dir_0"));

  Directory::removeTree("dir_0");
}

TEST(DirectoryTest, setCWD)
{
  Directory::create("dir_1");
  
  std::string baseDir = Path::makeAbsolute(getCurrDir());
  Directory::setCWD("dir_1");
  std::string cwd1 = Path::makeAbsolute(getCurrDir());    

  std::string cwd2 = Path::makeAbsolute(baseDir + Path::sep + std::string("dir_1"));
  ASSERT_EQ(cwd1, cwd2);
  
  Directory::setCWD(baseDir);
  ASSERT_EQ(baseDir, getCurrDir());
  Directory::removeTree("dir_1");
}

TEST(DirectoryTest, getCWD)
{
  ASSERT_EQ(getCurrDir(), Directory::getCWD());
}

TEST(DirectoryTest, RemoveTreeAndCreate)
{

  std::string p = Path::makeAbsolute(std::string("someDir"));
  std::string d = Path::join(p, "someSubDir");
  if (Path::exists(p))
    Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));
  ASSERT_THROW(Directory::create(d), exception); // nonrecursive create should fail
  Directory::create(d, false, true /* recursive */);
  ASSERT_TRUE(Path::exists(d));
  Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(d));
  ASSERT_TRUE(!Path::exists(p));
}


TEST(DirectoryTest, CopyTree)
{
  std::string p = Path::makeAbsolute(std::string("someDir"));
  std::string a = Path::join(p, "A");
  std::string b = Path::join(p, "B");
  
  if (Path::exists(p))
    Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));

  Directory::create(a, false, true /* recursive */);
  ASSERT_TRUE(Path::exists(a));

  Directory::create(b);
  ASSERT_TRUE(Path::exists(b));
  std::string src(Path::join(b, "1.txt"));
  if (Path::exists(src))
    Path::remove(src);
  ASSERT_TRUE(!Path::exists(src));

  {
    OFStream f(src.c_str());
    f << "12345";
    f.close();
  }
  ASSERT_TRUE(Path::exists(src));

  std::string dest = Path::join(a, "B", "1.txt");

  ASSERT_TRUE(!Directory::exists(Path::normalize(Path::join(a, "B"))));
    
  Directory::copyTree(b, a);
  ASSERT_TRUE(Directory::exists(Path::normalize(Path::join(a, "B"))));

  ASSERT_TRUE(Path::exists(dest));

  {
    std::string s;
    IFStream f(dest.c_str());
    f >> s;
    ASSERT_TRUE(s == "12345");
    f.close();
  }

  Directory::removeTree(p);
  ASSERT_TRUE(!Path::exists(p));
    
}

TEST(DirectoryTest, Iterator)
{
  if (Directory::exists("A"))
    Directory::removeTree("A");
  Directory::create("A");
  Directory::create("A" + sep + "B");
  Directory::create("A" + sep + "C");
  
  {
    Directory::Iterator di("A");
    Directory::Entry entry;
    Directory::Entry * e = nullptr;

    vector<string> subdirs;
    e = di.next(entry);
    ASSERT_TRUE(e != nullptr);
    ASSERT_TRUE(e->type == Directory::Entry::DIRECTORY);
    subdirs.push_back(e->path);
    string first = e->path;
    
    e = di.next(entry);
    ASSERT_TRUE(e != nullptr);
    ASSERT_TRUE(e->type == Directory::Entry::DIRECTORY);
    subdirs.push_back(e->path);
    
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
    ASSERT_TRUE(e->path == first);
  }  	

  // Cleanup test dirs
  ASSERT_TRUE(Path::exists("A"));
  Directory::removeTree("A");
  ASSERT_TRUE(!Path::exists("A"));
}

