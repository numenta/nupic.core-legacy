# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

# Creates ExternalProject for building yaml lib static library
#
# Exports:
#   LIB_STATIC_YAML_LOC: path to installed static yaml lib


set(yamllib_url "${REPOSITORY_DIR}/external/common/share/yaml/yaml-0.1.5.tar.gz")

# NOTE Yaml lib doesn't have an install target and leaves artifacts in build dir
set(yamllib_build_dir "${EP_BASE}/Build/YamlStaticLib")

# Export path to installed static yaml lib to parent
set(LIB_STATIC_YAML_LOC "${yamllib_build_dir}/${STATIC_PRE}yaml${STATIC_SUF}")

set(c_flags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")

# gcc v4.9 requires its own binutils-wrappers for LTO (flag -flto)
# fixes #981
IF(UNIX AND CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9" OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "4.9"))
	ExternalProject_Add(YamlStaticLib
    	URL ${yamllib_url}

    	UPDATE_COMMAND ""

    	# NOTE Yaml provides no rule for install
    	INSTALL_COMMAND ""

    	CMAKE_GENERATOR ${CMAKE_GENERATOR}

    	CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        	-DBUILD_SHARED_LIBS=OFF
        	-DCMAKE_C_FLAGS=${c_flags}
        	-DCMAKE_INSTALL_PREFIX=${yamllib_build_dir}
			-DCMAKE_AR=/usr/bin/gcc-ar
        	-DCMAKE_RANLIB=/usr/bin/gcc-ranlib
	)
ELSE()
	ExternalProject_Add(YamlStaticLib
    	URL ${yamllib_url}

    	UPDATE_COMMAND ""

    	# NOTE Yaml provides no rule for install
    	INSTALL_COMMAND ""

    	CMAKE_GENERATOR ${CMAKE_GENERATOR}

    	CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        	-DBUILD_SHARED_LIBS=OFF
        	-DCMAKE_C_FLAGS=${c_flags}
        	-DCMAKE_INSTALL_PREFIX=${yamllib_build_dir}
	)
ENDIF()
