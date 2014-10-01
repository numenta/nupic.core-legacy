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