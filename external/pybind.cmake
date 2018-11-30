##########################
#  Fetch pybind11 from Git
# This will download pybind11 at configure time, and add_subdirectory it. Then you're ready to call pybind11_add_module.
# See https://stackoverflow.com/questions/47027741/smart-way-to-cmake-a-project-using-pybind11-by-externalproject-add?rq=1
include(FetchContent)
FetchContent_Declare(
	pybind11
    FETCHCONTENT_BASE_DIR ${EP_BASE}
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()	
##########################
