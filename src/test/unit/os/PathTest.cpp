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
 * Implementation for Path test
 */

#include <nupic/os/Directory.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/FStream.hpp>
#include <nupic/utils/Log.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace nupic;


class PathTest : public ::testing::Test {
public:
  string testOutputDir_ = Path::makeAbsolute("TestEverything.out");

  PathTest() {
    // Create if it doesn't exist
    if (!Path::exists(testOutputDir_)) {
      std::cout << "Tester -- creating output directory " << std::string(testOutputDir_) << "\n";
      // will throw if unsuccessful. 
      Directory::create(string(testOutputDir_));
    } 
  }

  string fromTestOutputDir(const string& path) {
    Path testoutputpath(testOutputDir_);
    if (path != "")
      testoutputpath += path;
    return string(testoutputpath);
  }

};

// test static exists()
TEST_F(PathTest, exists)
{
}

TEST_F(PathTest, getParent)
{
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
  std::string g = "/a/b/c/g.ext";
  g = Path::getParent(g);
  EXPECT_STREQ("/a/b/c", g.c_str()) << "getParent1";

  g = Path::getParent(g);
  EXPECT_STREQ("/a/b", g.c_str()) << "getParent2";

  g = Path::getParent(g);
  EXPECT_STREQ("/a", g.c_str()) << "getParent3";

  g = Path::getParent(g);
  EXPECT_STREQ("/", g.c_str()) << "getParent4";

  g = Path::getParent(g);
  EXPECT_STREQ("/", g.c_str()) << "getParent5";
  
  // Parent should normalize first, to avoid parent(a/b/..)->(a/b)
  g = "/a/b/..";
  EXPECT_STREQ("/", Path::getParent(g).c_str()) << "getParent6";

  // getParent() of a relative directory may be a bit non-intuitive
  g = "a/b";
  EXPECT_STREQ("a", Path::getParent(g).c_str()) << "getParent7";

  g = "a";
  EXPECT_STREQ(".", Path::getParent(g).c_str()) << "getParent8";
  
  // getParent() of a relative directory above us should work
  g = "../../a";
  EXPECT_STREQ("../..", Path::getParent(g).c_str()) << "getParent9";

  g = ".";
  EXPECT_STREQ("..", Path::getParent(g).c_str()) << "getParent10";
  
#endif

  
  std::string x = Path::join("someDir", "X");
  x = Path::makeAbsolute(x);
  std::string y = Path::join(x, "Y");

  
  std::string parent = Path::getParent(y);
  ASSERT_TRUE(x == parent);

}

// test static getFilename()
TEST_F(PathTest, getFilename)

{
}

// test static getBasename()
TEST_F(PathTest, getBasename)
{
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
  EXPECT_STREQ("bar", Path::getBasename("/foo/bar").c_str()) << "basename1";
  EXPECT_STREQ("", Path::getBasename("/foo/bar/").c_str()) << "basename2";
  EXPECT_STREQ("bar.ext",
    Path::getBasename("/this is a long dir / foo$/bar.ext").c_str())
    << "basename3";
#endif
}

// test static getExtension()
TEST_F(PathTest, getExtension)
{
  string sep(Path::sep);
  std::string ext = Path::getExtension("abc" + sep + "def.ext");
  ASSERT_TRUE(ext == "ext");
}

// test static normalize()
TEST_F(PathTest, normalize)
{
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
  EXPECT_STREQ("/foo/bar", Path::normalize("//foo/quux/..//bar").c_str())
    << "normalize1";
  EXPECT_STREQ("/foo/contains a lot of spaces", 
       Path::normalize("///foo/a/b/c/../../d/../../contains a lot of spaces/g.tgz/..").c_str())
    << "normalize2";
  EXPECT_STREQ("../..", Path::normalize("../foo/../..").c_str()) 
    << "normalize3";
  EXPECT_STREQ("/", Path::normalize("/../..").c_str()) << "normalize4";
#endif         

}

// test static makeAbsolute()
TEST_F(PathTest, makeAbsolute)
{
}

// test static split()
TEST_F(PathTest, split)
{
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
  Path::StringVec sv;
  sv = Path::split("/foo/bar");
  ASSERT_EQ(3U, sv.size()) << "split1 size";
  if (sv.size() == 3) {
    ASSERT_EQ(sv[0], "/") << "split1.1";
    ASSERT_EQ(sv[1], "foo") << "split1.2";
    ASSERT_EQ(sv[2], "bar") << "split1.3";
  }
  EXPECT_STREQ("/foo/bar", Path::join(sv.begin(), sv.end()).c_str()) << "split1.4";

  sv = Path::split("foo/bar");
  ASSERT_EQ(2U, sv.size()) << "split2 size";
  if (sv.size() == 2) 
  {
    ASSERT_EQ(sv[0], "foo") << "split2.2";
    ASSERT_EQ(sv[1], "bar") << "split2.3";
  }
  EXPECT_STREQ("foo/bar", Path::join(sv.begin(), sv.end()).c_str())
    << "split2.3";

  sv = Path::split("foo//bar/");
  ASSERT_EQ(2U, sv.size()) << "split3 size";
  if (sv.size() == 2) 
  {
    ASSERT_EQ(sv[0], "foo") << "split3.2";
    ASSERT_EQ(sv[1], "bar") << "split3.3";
  }
  EXPECT_STREQ("foo/bar", Path::join(sv.begin(), sv.end()).c_str())
    << "split3.4";

#endif 


}

// test static join()
TEST_F(PathTest, join)
{
}

// test static remove()
TEST_F(PathTest, remove)
{
}

// test static rename()
TEST_F(PathTest, rename)
{
}

// test static copy()
TEST_F(PathTest, copy)
{
  {
    OFStream f("a.txt");
    f << "12345";
  }

  {
    std::string s;
    IFStream f("a.txt");
    f >> s;
    ASSERT_TRUE(s == "12345");
  }
  
  {
    if (Path::exists("b.txt"))
      Path::remove("b.txt");
    ASSERT_TRUE(!Path::exists("b.txt"));
    Path::copy("a.txt", "b.txt");
    ASSERT_TRUE(Path::exists("b.txt"));
    std::string s;
    IFStream f("b.txt");
    f >> s;
    ASSERT_TRUE(s == "12345");
  }
  
  Path::remove("a.txt");
  Path::remove("b.txt");
  ASSERT_TRUE(!Path::exists("a.txt"));
  ASSERT_TRUE(!Path::exists("b.txt"));
}    

// test static copy() in temp directory
TEST_F(PathTest, copyInTemp)
{
  {
    OFStream f("a.txt");
    f << "12345";
  }

  {
    std::string s;
    IFStream f("a.txt");
    f >> s;
    ASSERT_TRUE(s == "12345");
  }
  
  string destination = fromTestOutputDir("pathtest_dir");
  {
    destination += "b.txt";
    if (Path::exists(destination))
      Path::remove(destination);
    ASSERT_FALSE(Path::exists(destination));
    Path::copy("a.txt", destination);
    ASSERT_TRUE(Path::exists(destination));
    std::string s;
    IFStream f(destination.c_str());
    f >> s;
    ASSERT_TRUE(s == "12345");
  }
  
  Path::remove("a.txt");
  Path::remove(destination);
  ASSERT_TRUE(!Path::exists("a.txt"));
  ASSERT_TRUE(!Path::exists(destination));
}    

//test static isRootdir()
TEST_F(PathTest, isRootDir)
{
}

//test static isAbsolute()
TEST_F(PathTest, isAbsolute)
{
#if defined(NTA_OS_WINDOWS)
  ASSERT_TRUE(Path::isAbsolute("c:"));
  ASSERT_TRUE(Path::isAbsolute("c:\\"));
  ASSERT_TRUE(Path::isAbsolute("c:\\foo\\"));
  ASSERT_TRUE(Path::isAbsolute("c:\\foo\\bar"));    
  
  ASSERT_TRUE(Path::isAbsolute("\\\\foo"));    
  ASSERT_TRUE(Path::isAbsolute("\\\\foo\\"));    
  ASSERT_TRUE(Path::isAbsolute("\\\\foo\\bar"));
  ASSERT_TRUE(Path::isAbsolute("\\\\foo\\bar\\baz"));
     
  ASSERT_TRUE(!Path::isAbsolute("foo"));        
  ASSERT_TRUE(!Path::isAbsolute("foo\\bar"));        
  ASSERT_TRUE(!Path::isAbsolute("\\"));
  ASSERT_TRUE(!Path::isAbsolute("\\\\"));
  ASSERT_TRUE(!Path::isAbsolute("\\foo"));
#else
  ASSERT_TRUE(Path::isAbsolute("/"));
  ASSERT_TRUE(Path::isAbsolute("/foo"));
  ASSERT_TRUE(Path::isAbsolute("/foo/"));
  ASSERT_TRUE(Path::isAbsolute("/foo/bar"));    
      
  ASSERT_TRUE(!Path::isAbsolute("foo"));        
  ASSERT_TRUE(!Path::isAbsolute("foo/bar"));        
#endif 
}

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
