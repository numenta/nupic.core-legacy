cmake_minimum_required (VERSION 2.8.11)

project (nupic_core_example)

set(LIB_NUPIC_CORE_LIB nupic-core)
add_library(${LIB_NUPIC_CORE_LIB} STATIC IMPORTED)
set_property(TARGET ${LIB_NUPIC_CORE_LIB} PROPERTY IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/../../lib/libnupic_core.lib")

add_executable (helloregions_shared helloregions.cpp)
include_directories("../../include")
set(LIBS ${LIBS} ${LIB_NUPIC_CORE_LIB}) 
target_link_libraries(helloregions_shared ${LIBS})
add_definitions(-DNTA_PLATFORM_win32)
set(LIB_NUPIC_CORE_DLL_PATH "${PROJECT_SOURCE_DIR}/../../bin/libnupic_core.dll")
string(REPLACE "/" "\\" LIB_NUPIC_CORE_DLL_PATH ${LIB_NUPIC_CORE_DLL_PATH})
set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>")
string(REPLACE "/" "\\" OUTPUT_PATH ${OUTPUT_PATH})
set(DATA_PATH "${PROJECT_SOURCE_DIR}/data.csv")
string(REPLACE "/" "\\" DATA_PATH ${DATA_PATH})

ADD_CUSTOM_COMMAND(
    TARGET helloregions_shared
    POST_BUILD COMMAND copy "${LIB_NUPIC_CORE_DLL_PATH}" "${OUTPUT_PATH}"
    POST_BUILD COMMAND copy "${DATA_PATH}" "${OUTPUT_PATH}"
)