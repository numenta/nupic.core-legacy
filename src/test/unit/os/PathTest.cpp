/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Implementation for Path test
 */

#include <nupic/os/Path.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/FStream.hpp>
#include <nupic/utils/Log.hpp>

#include "PathTest.hpp"

using namespace std;
using namespace nupic;

void PathTest::RunTests()
{
  std::string sep(Path::sep);
  
  // test static exists()
  {
  }

  // test static getParent()
  {
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
    std::string g = "/a/b/c/g.ext";
    g = Path::getParent(g);
    TESTEQUAL2_STR("getParent1", "/a/b/c", g.c_str());

    g = Path::getParent(g);
    TESTEQUAL2_STR("getParent2", "/a/b", g.c_str());

    g = Path::getParent(g);
    TESTEQUAL2_STR("getParent3", "/a", g.c_str());

    g = Path::getParent(g);
    TESTEQUAL2_STR("getParent4", "/", g.c_str());

    g = Path::getParent(g);
    TESTEQUAL2_STR("getParent5", "/", g.c_str());
    
    // Parent should normalize first, to avoid parent(a/b/..)->(a/b)
    g = "/a/b/..";
    TESTEQUAL2_STR("getParent6", "/", Path::getParent(g).c_str());

    // getParent() of a relative directory may be a bit non-intuitive
    g = "a/b";
    TESTEQUAL2_STR("getParent7", "a", Path::getParent(g).c_str());

    g = "a";
    TESTEQUAL2_STR("getParent8", ".", Path::getParent(g).c_str());
    
    // getParent() of a relative directory above us should work
    g = "../../a";
    TESTEQUAL2_STR("getParent9", "../..", Path::getParent(g).c_str());

    g = ".";
    TESTEQUAL2_STR("getParent10", "..", Path::getParent(g).c_str());
    
#endif

    
    std::string x = Path::join("someDir", "X");
    x = Path::makeAbsolute(x);
    std::string y = Path::join(x, "Y");

    
    std::string parent = Path::getParent(y);
    TEST(x == parent);

  }

  // test static getFilename()
  {
  }

  // test static getBasename()
  {
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
    TESTEQUAL2_STR("basename1", "bar", Path::getBasename("/foo/bar").c_str());
    TESTEQUAL2_STR("basename2", "", Path::getBasename("/foo/bar/").c_str());
    TESTEQUAL2_STR("basename3", "bar.ext", Path::getBasename("/this is a long dir / foo$/bar.ext").c_str());
#endif
  }
  
  // test static getExtension()
  {
    std::string ext = Path::getExtension("abc" + sep + "def.ext");
    TEST(ext == "ext");
  }

  // test static normalize()
  {
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
    TESTEQUAL2_STR("normalize1", "/foo/bar", Path::normalize("//foo/quux/..//bar").c_str());
    TESTEQUAL2_STR("normalize2", "/foo/contains a lot of spaces", 
         Path::normalize("///foo/a/b/c/../../d/../../contains a lot of spaces/g.tgz/..").c_str());
    TESTEQUAL2_STR("normalize3", "../..", Path::normalize("../foo/../..").c_str());
    TESTEQUAL2_STR("normalize4", "/", Path::normalize("/../..").c_str());
#endif         

  }

  // test static makeAbsolute()
  {
  }

  // test static split()
  {
#if defined(NTA_OS_WINDOWS)
// no tests defined
#else
    Path::StringVec sv;
    sv = Path::split("/foo/bar");
    TESTEQUAL2("split1 size", 3U, sv.size());
    if (sv.size() == 3) {
      TESTEQUAL2("split1.1", sv[0], "/");
      TESTEQUAL2("split1.2", sv[1], "foo");
      TESTEQUAL2("split1.3", sv[2], "bar");
    }
    TESTEQUAL2_STR("split1.4", "/foo/bar", Path::join(sv.begin(), sv.end()).c_str());

    sv = Path::split("foo/bar");
    TESTEQUAL2("split2 size", 2U, sv.size());
    if (sv.size() == 2) 
    {
      TESTEQUAL2("split2.2", sv[0], "foo");
      TESTEQUAL2("split2.3", sv[1], "bar");
    }
    TESTEQUAL2_STR("split2.3", "foo/bar", Path::join(sv.begin(), sv.end()).c_str());

    sv = Path::split("foo//bar/");
    TESTEQUAL2("split3 size", 2U, sv.size());
    if (sv.size() == 2) 
    {
      TESTEQUAL2("split3.2", sv[0], "foo");
      TESTEQUAL2("split3.3", sv[1], "bar");
    }
    TESTEQUAL2_STR("split3.4", "foo/bar", Path::join(sv.begin(), sv.end()).c_str());

#endif 


  }

  // test static join()
  {
  }

  // test static remove()
  {
  }

  // test static rename()
  {
  }
  
  // test static copy()
  {
    {
      OFStream f("a.txt");
      f << "12345";
    }

    {
      std::string s;
      IFStream f("a.txt");
      f >> s;
      TEST(s == "12345");
    }
    
    {
      if (Path::exists("b.txt"))
        Path::remove("b.txt");
      TEST(!Path::exists("b.txt"));
      Path::copy("a.txt", "b.txt");
      TEST(Path::exists("b.txt"));
      std::string s;
      IFStream f("b.txt");
      f >> s;
      TEST(s == "12345");
    }
    
    Path::remove("a.txt");
    Path::remove("b.txt");
    TEST(!Path::exists("a.txt"));
    TEST(!Path::exists("b.txt"));
  }    

  // test static copy() in temp directory
  {
    {
      OFStream f("a.txt");
      f << "12345";
    }

    {
      std::string s;
      IFStream f("a.txt");
      f >> s;
      TEST(s == "12345");
    }
    
    string destination = fromTestOutputDir("pathtest_dir");
    {
      destination += "b.txt";
      if (Path::exists(destination))
        Path::remove(destination);
      TEST(!Path::exists(destination));
      Path::copy("a.txt", destination);
      TEST(Path::exists(destination));
      std::string s;
      IFStream f(destination.c_str());
      f >> s;
      TEST(s == "12345");
    }
    
    Path::remove("a.txt");
    Path::remove(destination);
    TEST(!Path::exists("a.txt"));
    TEST(!Path::exists(destination));
  }    
  
  //test static isRootdir()
  {
  }

  //test static isAbsolute()
  {
  #if defined(NTA_OS_WINDOWS)
    TEST(Path::isAbsolute("c:"));
    TEST(Path::isAbsolute("c:\\"));
    TEST(Path::isAbsolute("c:\\foo\\"));
    TEST(Path::isAbsolute("c:\\foo\\bar"));    
    
    TEST(Path::isAbsolute("\\\\foo"));    
    TEST(Path::isAbsolute("\\\\foo\\"));    
    TEST(Path::isAbsolute("\\\\foo\\bar"));
    TEST(Path::isAbsolute("\\\\foo\\bar\\baz"));
       
    TEST(!Path::isAbsolute("foo"));        
    TEST(!Path::isAbsolute("foo\\bar"));        
    TEST(!Path::isAbsolute("\\"));
    TEST(!Path::isAbsolute("\\\\"));
    TEST(!Path::isAbsolute("\\foo"));
  #else
    TEST(Path::isAbsolute("/"));
    TEST(Path::isAbsolute("/foo"));
    TEST(Path::isAbsolute("/foo/"));
    TEST(Path::isAbsolute("/foo/bar"));    
        
    TEST(!Path::isAbsolute("foo"));        
    TEST(!Path::isAbsolute("foo/bar"));        
  #endif 
  }

  { 
    // test static getExecutablePath
    std::string path = Path::getExecutablePath();
    std::cout << "Executable path: '" << path << "'\n";
    TEST(Path::exists(path));

    std::string basename = Path::getBasename(path);
#if defined(NTA_OS_WINDOWS)
    TESTEQUAL2_STR("basename should be unit_tests", basename.c_str(), "unit_tests.exe");
#else
    TESTEQUAL2_STR("basename should be unit_tests", basename.c_str(), "unit_tests");
#endif
  }    


}

