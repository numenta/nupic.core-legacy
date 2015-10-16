# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# -----------------------------------------------------------------------------

option(SOURCE_SWIG "Build SWIG from source even if it is found." OFF)

if (NOT ${SOURCE_SWIG})
  find_package(SWIG)
endif ()

if (NOT SWIG_FOUND)
  # Build SWIG from source.
  ExternalProject_Add(
    Swig
    GIT_REPOSITORY https://github.com/swig/swig.git
    GIT_TAG rel-3.0.2
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
        ./configure --prefix=${EP_PREFIX}/Install
  )
else()
  # Create a dummy target to depend on.
  add_custom_target(Swig)
endif()
