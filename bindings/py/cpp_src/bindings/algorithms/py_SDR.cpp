
#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

#include <nupic/ntypes/Sdr.hpp>

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_SDR(py::module& m)
    {
        py::class_<SDR> py_SDR(m, "SDR");

    }
}
