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


# figures out the version from git and returns Version, Major, Minor, Patch
#

function(get_versions version major minor patch)
	find_package(Git)
	if(GIT_FOUND)
		execute_process(COMMAND "${GIT_EXECUTABLE}" log --pretty=format:'%h' -n 1
	                OUTPUT_VARIABLE GIT_REV
			OUTPUT_STRIP_TRAILING_WHITESPACE
	                ERROR_QUIET)
	endif()
	# Check whether we got any revision (which isn't
	# always the case, e.g. when someone downloaded a zip
	# file from Github instead of a checkout
	if ("${GIT_REV}" STREQUAL "")
	    set(GIT_TAG "0.0.0")
	    set(GIT_BRANCH "No branch")
	    # use the version already in the VERSION file.
	    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
	       file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION m_m_p_)
	    else()
	       set(m_m_p_ "0.0.0")
	    endif()
	    set(${m_version} ${m_m_p_} PARENT_SCOPE)	    
	else()
	    string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
	    
	    # see if there as a tag set on this PR
	    execute_process(
	        COMMAND "${GIT_EXECUTABLE}" describe --exact-match --tags
	        OUTPUT_VARIABLE GIT_TAG 
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET)
	    execute_process(
	        COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
	        OUTPUT_VARIABLE GIT_BRANCH
		OUTPUT_STRIP_TRAILING_WHITESPACE)
        
    
	    if ("${GIT_TAG}" STREQUAL "")
	        execute_process(
	            COMMAND "${GIT_EXECUTABLE}" describe --tags
	            OUTPUT_VARIABLE GIT_TAG 
		    OUTPUT_STRIP_TRAILING_WHITESPACE
		    ERROR_QUIET)
	    endif()
    
	    if ("${GIT_TAG}" STREQUAL "")
	        set(m_version ${GIT_REV})
		set(${version} ${m_version} PARENT_SCOPE)
		set(${major}   0 PARENT_SCOPE)
		set(${minor}   0 PARENT_SCOPE)
		set(${patch}   0 PARENT_SCOPE)
		return()
	    else()
	        set(m_version ${GIT_TAG})
	    endif()
	endif()

   
	string(REGEX REPLACE "^([vV])([0-9]*)([.][0-9]*[.][0-9]*-?.*)$" "\\2" numbers ${m_version} )
	set(m_major ${numbers})
	string(REGEX REPLACE "^([vV][0-9]*[.])([0-9]*)([.][0-9]*-?.*)$" "\\2" numbers ${m_version} )
	set(m_minor ${numbers})
	string(REGEX REPLACE "^([vV][0-9]*[.][0-9]*[.])([0-9]*)(-?.*)$" "\\2" numbers ${m_version} )
	set(m_patch ${numbers})
	


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