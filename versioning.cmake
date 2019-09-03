# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Numenta, Inc.
#
# Author: David Keeney, 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# -----------------------------------------------------------------------------


# This function figures out the version from git and returns Version, Major, Minor, Patch
#
#  This function will query git to find the most recent tag by calling 
#       git describe --tags
#  If there is a tag on the PR then this will be an exact match and this 
#  determins the version.  If there have been commits
#  since the tag it may contain additional info relating to the commit.
#
#  Once we have the version, we parse out the Major, Minor, and Patch components
#  and we write the version into the VERSION file if its different.
#  The components VERSION, MAJOR, MINOR, and PATCH are returned to
#  the top level CMakeLists.txt file.  The VERSION file's value is also
#  integrated into the created version.h file which is part of the 
#  includes folder in the GitHub binary distribution.
#
#  These four components are used to create the GitHub release package during
#  packaging.   The value in VERSION file will be used by setup.py to set
#  the version when it creates the wheel file which is the package for PYPI.
#
#  If someone should try to build without having cloned the repository
#  (just downloaded a tag.gz or zip), it will just use the current value 
#  in the VERSION file.

function(get_versions version major minor patch)
    # do we have Git installed and are we in a repository?
    find_package(Git)
    if(GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" log --pretty=format:'%h' -n 1
            OUTPUT_VARIABLE GIT_REV
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    # Check whether we got any revision; which isn't
    # always the case, e.g. when someone downloaded a tar.gz or zip
    # file from Github instead of a checkout...
    # or maybe git is not even installed.
    if ("${GIT_REV}" STREQUAL "")
        # use the version already in the VERSION file.
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
           message(STATUS "NOTE: USING version from VERSION file. No git.")
           file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION GIT_TAG)
        endif()
    else()
        
        # see if there as a tag set on this branch
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe --exact-match --tags
            OUTPUT_VARIABLE GIT_TAG 
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    
        if ("${GIT_TAG}" STREQUAL "")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" describe --tags
                OUTPUT_VARIABLE GIT_TAG 
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
        endif()
        
        if ("${GIT_TAG}" STREQUAL "")
           # use the version already in the VERSION file.
           # This could happen if not enough of the repository is downloaded.
           # i.e. use "git fetch --depth=200"
           if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
              message(STATUS "NOTE: USING version from VERSION file. No git.")
              file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION GIT_TAG)
           endif()
        endif()
    endif()
    
    if ("${GIT_TAG}" STREQUAL "")
        # even the VERSION file must have been empty...should not happen.
        set(m_version ${GIT_REV})
        set(${version} ${m_version} PARENT_SCOPE)
        set(${major}   0 PARENT_SCOPE)
        set(${minor}   0 PARENT_SCOPE)
        set(${patch}   0 PARENT_SCOPE)
        return()
    endif()
    
    string(REGEX MATCH "^[vV][0-9]*[.][0-9]*[.][0-9]*-?.*$" matched ${GIT_TAG})
    if ("${matched}" STREQUAL "")
        # did not match vM.M.P-??? pattern
        set(m_version ${GIT_TAG})
        set(${version} ${m_version} PARENT_SCOPE)
        set(${major}   0 PARENT_SCOPE)
        set(${minor}   0 PARENT_SCOPE)
        set(${patch}   0 PARENT_SCOPE)
        return()
    endif()        

    set(m_version ${GIT_TAG})

    message(STATUS "Full Version from Git: ${m_version}")
   
    string(REGEX REPLACE "^([vV])([0-9]*)([.][0-9]*[.][0-9]*-?.*)$" "\\2" numbers ${m_version} )
    set(m_major ${numbers})
    string(REGEX REPLACE "^([vV][0-9]*[.])([0-9]*)([.][0-9]*-?.*)$" "\\2" numbers ${m_version} )
    set(m_minor ${numbers})
    string(REGEX REPLACE "^([vV][0-9]*[.][0-9]*[.])([0-9]*)(-?.*)$" "\\2" numbers ${m_version} )
    set(m_patch ${numbers})
    
    # limiting ourselves to v Major.Minor.Patch format for version.
    set(m_version "v${m_major}.${m_minor}.${m_patch}")

    # Write it to the file VERSION if it is different.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
        file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION m_version_)
    else()
        set(m_version_ "")
    endif()

    if (NOT "${m_version}" STREQUAL "${m_version_}")
        file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/VERSION "${m_version}")
    endif()
    
    # return the arguments
    set(${version} ${m_version} PARENT_SCOPE)
    set(${major}   ${m_major} PARENT_SCOPE)
    set(${minor}   ${m_minor} PARENT_SCOPE)
    set(${patch}   ${m_patch} PARENT_SCOPE)
endfunction()