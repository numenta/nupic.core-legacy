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
 * Implementation for Path test
 *
 * NOTE: EXPECT ALL OUTPUT TO BE PLACED IN TestOutputDir FOLDER.
 *       The Test jig class PathTest will delete everything in destructor.
 */

#include <gtest/gtest.h>

#include <htm/os/Directory.hpp>
#include <htm/os/Path.hpp>
#include <htm/utils/Log.hpp>

namespace testing {
    
using namespace std;
using namespace htm;

#define VERBOSE if(verbose)std::cerr << "[          ] "
static bool verbose = false;  // turn this on to print extra stuff for debugging the test.

class PathTest : public ::testing::Test {
public:
  std::string testOutputDir_ = Path::makeAbsolute("TestOutputDir");
  std::string origCwd_;

  PathTest() {
    // Create output directory
    if (Path::exists(testOutputDir_)) {
      Directory::removeTree(testOutputDir_);
    }
    VERBOSE << "New testbed directory: " << testOutputDir_ << std::endl;
    // will throw if unsuccessful.
    Directory::create(testOutputDir_);

    origCwd_ = Directory::getCWD();   // remember the original current working directory.
    Directory::setCWD(insideTestOutputDir(""));
  }

  ~PathTest() {
    Directory::setCWD(origCwd_);  // restore CWD
    if (Path::exists(testOutputDir_)) {
      Directory::removeTree(testOutputDir_);
    }
    VERBOSE << "testbed taken down." << std::endl;
  }

  // create an absolute path that is inside the testOutputDir
  string insideTestOutputDir(const string& path)
  {
    string testoutputpath = testOutputDir_ + Path::sep + path;
    return testoutputpath;
  }
};

//test static isRootdir()
TEST_F(PathTest, isRootDir)
{
#if defined(NTA_OS_WINDOWS)
  std::string cwd = Directory::getCWD();
  if (cwd.length() > 3) cwd = cwd.substr(0, 3);
  ASSERT_TRUE(Path::isRootdir(cwd));
#else
  ASSERT_TRUE(Path::isRootdir("/"));
#endif
}

//test static isAbsolute()
TEST_F(PathTest, isAbsolute)
{
#if defined(NTA_OS_WINDOWS)
  EXPECT_TRUE(Path::isAbsolute("c:"));
  EXPECT_TRUE(Path::isAbsolute("c:/"));
  EXPECT_TRUE(Path::isAbsolute("c:\\"));
  EXPECT_TRUE(Path::isAbsolute("c:\\foo\\"));
  EXPECT_TRUE(Path::isAbsolute("c:\\foo\\bar"));

  EXPECT_TRUE(Path::isAbsolute("\\\\foo"));
  EXPECT_TRUE(Path::isAbsolute("\\\\foo\\"));
  EXPECT_TRUE(Path::isAbsolute("\\\\foo\\bar"));
  EXPECT_TRUE(Path::isAbsolute("\\\\foo\\bar\\baz"));

  EXPECT_TRUE(!Path::isAbsolute("foo"));
  EXPECT_TRUE(!Path::isAbsolute("foo\\bar"));
  EXPECT_TRUE(!Path::isAbsolute("/"));
  EXPECT_TRUE(!Path::isAbsolute("\\"));
  EXPECT_TRUE(!Path::isAbsolute("\\\\"));
  EXPECT_TRUE(!Path::isAbsolute("\\foo"));
#else
  EXPECT_TRUE(Path::isAbsolute("/"));
  EXPECT_TRUE(Path::isAbsolute("/foo"));
  EXPECT_TRUE(Path::isAbsolute("/foo/"));
  EXPECT_TRUE(Path::isAbsolute("/foo/bar"));

  EXPECT_TRUE(!Path::isAbsolute("foo"));
  EXPECT_TRUE(!Path::isAbsolute("foo/bar"));
#endif
}



TEST_F(PathTest, getParent)
{
  // Note: Path compares are normalized Lexical compares
  //       so they will match on both windows or linux.
  std::string g = "/a/b\\c/g.ext";
  g = Path::getParent(g);
  EXPECT_FALSE(Path("/a/b/c1") == Path(g)) << "getParent1 negative";
  EXPECT_TRUE(Path("/a/b/c") == Path(g)) << "getParent1";

  g = Path::getParent(g);
  EXPECT_TRUE(Path("/a/b") == g) << "getParent2";

  g = Path::getParent(g);
  EXPECT_TRUE(Path("/a") == g) << "getParent3";

  g = Path::getParent(g);
  EXPECT_TRUE(Path("/") == g) << "getParent4";

  g = Path::getParent(g);
  EXPECT_TRUE(Path("/") == g) << "getParent5";

  // Parent should normalize first, to avoid parent(a/b/..)->(a/b)
  g = "/a/b/..";
  EXPECT_TRUE(Path("/") == Path::getParent(g)) << "getParent6";

  // getParent() of a relative directory may be a bit non-intuitive
  g = "a/b";
  EXPECT_TRUE(Path("a") == Path::getParent(g)) << "getParent7a";

  g = "a/b/";
  EXPECT_TRUE(Path("a") == Path::getParent(g)) << "getParent7b";

  g = "a/";
  EXPECT_TRUE(Path(".") == Path::getParent(g)) << "getParent8a";

  g = "a";
  EXPECT_TRUE(Path(".") == Path::getParent(g)) << "getParent8b";

  // getParent() of a relative directory above us should work
  g = "../../a";
  EXPECT_TRUE(Path("../..") == Path::getParent(g)) << "getParent9";

  g = ".";
  EXPECT_TRUE(Path("..") == Path::getParent(g)) << "getParent10";

  g = "..";
  EXPECT_TRUE(Path("../..") == Path::getParent(g)) << "getParent11";

  g = "../..";
  EXPECT_TRUE(Path("../../..") == Path::getParent(g)) << "getParent11";

  g = "./a";
  EXPECT_TRUE(Path(".") == Path::getParent(g)) << "getParent12";

  // check absolute paths on windows
  std::string x = std::string("someDir") + Path::sep + "X";
  x = Path::makeAbsolute(x);
  std::string y = x + std::string(Path::sep) + "Y";
  std::string parent = Path::getParent(y);
  ASSERT_TRUE(Path(x) == Path(parent));

}


// test static getBasename()
TEST_F(PathTest, getBasename)
{
  Path a("A/1.txt");
  EXPECT_STREQ(a.getBasename().c_str(), "1.txt");

  EXPECT_TRUE(Path("bar") == Path::getBasename("/foo/bar")) << "basename1";
  EXPECT_TRUE(Path(".") == Path::getBasename("/foo/bar/")) << "basename2";
  EXPECT_TRUE(Path("bar.ext") == Path::getBasename("/this is a long dir / foo$/bar.ext")) << "basename3";
}

// test static getExtension()
TEST_F(PathTest, getExtension)
{
  std::string ext = Path::getExtension("abc" + string(Path::sep) + "def.ext");
  EXPECT_TRUE(ext == "ext");

  ext = Path::getExtension("abc/def.ext.zip");
  EXPECT_TRUE(ext == "zip");
}

// test static normalize()
TEST_F(PathTest, normalize)
{
  EXPECT_TRUE(Path("//foo/bar") == Path::normalize("//foo/quux/..//bar"))
    << "normalize1";
  EXPECT_TRUE(Path("///foo/contains a lot of spaces") ==
       Path::normalize("   ///foo/a/b/c/../../d/../../ contains a lot of spaces/g.tgz/.. "))
    << "normalize2";
  EXPECT_TRUE(Path("../..") == Path::normalize("../foo/../.."))
    << "normalize3";
 // EXPECT_TRUE(Path("/") == Path::normalize("/../..")) << "normalize4";

}
// test static makeAbsolute()
TEST_F(PathTest, makeAbsolute)
{
  std::string g = "this/path.txt";
  Path p1(Directory::getCWD() + Path::sep + g);
  Path p2 = Path::makeAbsolute(g);
  EXPECT_TRUE(p1 == p2);
}


// test static remove()
TEST_F(PathTest, remove)
{
  if (Path::exists(insideTestOutputDir("A")) )
    Directory::removeTree(insideTestOutputDir("A"));
  Directory::create(insideTestOutputDir("A"));
  ASSERT_TRUE(Path::exists(testOutputDir_ + Path::sep + string("A")));
  ASSERT_TRUE(Path::exists("A"));  // the CWD should also be the testOutputDir.

  std::string dest = insideTestOutputDir("A/1.txt");
  Path::write_all(dest, "12345");
  ASSERT_TRUE(Path::exists(dest));
  Path::remove(dest);
  ASSERT_FALSE(Path::exists(dest));
  Path::remove(insideTestOutputDir("A"));
  ASSERT_FALSE(Path::exists(insideTestOutputDir("A")));
  ASSERT_TRUE(Directory::empty("."));   // make sure we cleaned up
}

// test static rename()
TEST_F(PathTest, rename)
{
  if (Path::exists("A"))
    Directory::removeTree("A");
  Directory::create("A");

  std::string src = "A/1.txt";
  std::string dest1 = "A/2.txt";
  std::string dest2 = "A/B/2.txt";
  std::string dest3 = "C/B/2.txt";
  Path::write_all(src, "12345");
  EXPECT_TRUE(Path::exists(src));
  Path::rename(src, dest1);
  EXPECT_FALSE(Path::exists(src));
  EXPECT_TRUE(Path::exists(dest1)) << "File did not get moved to A/2.txt";
  Directory::create("A/B");
  Path::rename(dest1, dest2);
  EXPECT_FALSE(Path::exists(dest1));
  EXPECT_TRUE(Path::exists(dest2)) << "File did not get moved to A/B/2.txt";
  Path::rename("A", "C");
  EXPECT_FALSE(Path::exists("A"));
  EXPECT_TRUE(Path::exists("C"));
  EXPECT_TRUE(Path::exists(dest3.c_str())) << "File did not get moved to C/B/2.txt";
  EXPECT_STREQ(Path::read_all(dest3).c_str(), "12345") << "Content of moved file not as expected.";
  Directory::removeTree("C");
  ASSERT_TRUE(Directory::empty("."));   // make sure we cleaned up
}

// test static copy()
TEST_F(PathTest, copyFile)
{
  Path::write_all("a.txt", "12345");
  ASSERT_STREQ(Path::read_all("a.txt").c_str(), "12345") << "Confirm contents of source file.";
  {
    if (Path::exists("b.txt"))
      Path::remove("b.txt");
    ASSERT_TRUE(!Path::exists("b.txt"));
    Path::copy("a.txt", "b.txt");
    ASSERT_TRUE(Path::exists("b.txt"));
    ASSERT_TRUE(Path::exists("a.txt"));
    ASSERT_STREQ(Path::read_all("b.txt").c_str(), "12345") << "Confirm contents of copied file.";
  }

  Path::remove("a.txt");
  Path::remove("b.txt");
  ASSERT_TRUE(!Path::exists("a.txt"));
  ASSERT_TRUE(!Path::exists("b.txt"));
  ASSERT_TRUE(Directory::empty("."));   // make sure we cleaned up
}

// test static copy() to directory
TEST_F(PathTest, copyFileToDir)
{
  Path::write_all("a.txt", "12345");
  ASSERT_STREQ(Path::read_all("a.txt").c_str(), "12345") << "Confirm contents of source file.";

  string destination = insideTestOutputDir("AA/b.txt");
  {
    if (Path::exists(destination))
      Path::remove(destination);
    ASSERT_FALSE(Path::exists(destination));
    Path::copy("a.txt", destination);
    ASSERT_TRUE(Path::exists(destination));
    ASSERT_STREQ(Path::read_all(destination).c_str(), "12345") << "Confirm contents of destination file.";
  }

  Path::remove("a.txt");
  Path::remove(Path::getParent(destination));
  ASSERT_TRUE(!Path::exists("a.txt"));
  ASSERT_TRUE(!Path::exists(destination));
  ASSERT_TRUE(Directory::empty("."));   // make sure we cleaned up
}

// test static copy() directory to directory
TEST_F(PathTest, copyDirToDir)
{
  Directory::create("A/B", false, true);
  {
    Path::write_all("A/a.txt", "01234");
    Path::write_all("A/b.txt", "56789");
    Path::write_all("A/B/c.txt", "abcde");
    Path::write_all("A/B/d.txt", "fghij");
  }

  Path::copy("A", "E");

  {
    ASSERT_TRUE(Path::exists("A/a.txt")) << "Source file still exists.";
    ASSERT_TRUE(Path::exists("E/a.txt"));
    ASSERT_STREQ(Path::read_all("E/a.txt").c_str(), "01234");
    ASSERT_STREQ(Path::read_all("E/b.txt").c_str(), "56789");
    ASSERT_TRUE(Path::exists("E/B"));
    ASSERT_STREQ(Path::read_all("E/B/c.txt").c_str(), "abcde");
    ASSERT_STREQ(Path::read_all("E/B/d.txt").c_str(), "fghij");
  }
  // copy again with overwrite
  Path::write_all("A/a.txt", "01234xxx");
  Path::copy("A", "E");

  {
    ASSERT_TRUE(Path::exists("A/a.txt")) << "Source file still exists.";
    ASSERT_TRUE(Path::exists("E/a.txt"));
    ASSERT_STREQ(Path::read_all("E/a.txt").c_str(), "01234xxx");
    ASSERT_STREQ(Path::read_all("E/b.txt").c_str(), "56789");
    ASSERT_TRUE(Path::exists("E/B"));
    ASSERT_STREQ(Path::read_all("E/B/c.txt").c_str(), "abcde");
    ASSERT_STREQ(Path::read_all("E/B/d.txt").c_str(), "fghij");
  }

  Path::remove("A");
  Path::remove("E");
  ASSERT_TRUE(Directory::empty("."));   // make sure we cleaned up
}

/******* removing this function until we find we really need it *****
TEST_F(PathTest, getExecutablePath)
{
  // test static getExecutablePath
  std::string path = Path::getExecutablePath();
  std::cout << "Executable path: '" << path << "'\n";
  ASSERT_TRUE(Path::exists(path));

  std::string basename = Path::getBasename(path);
#if defined(NTA_OS_WINDOWS)
  EXPECT_STREQ(basename.c_str(), "unit_tests.exe")
     << "basename should be unit_tests";
#else
  EXPECT_STREQ(basename.c_str(), "unit_tests")
    << "basename should be unit_tests";
#endif
}

**************************************************************/
}
