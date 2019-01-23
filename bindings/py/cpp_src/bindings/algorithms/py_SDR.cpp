
#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/ntypes/Sdr.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

using namespace nupic;

// TODO: docstrings everywhere!
namespace nupic_ext
{
    void init_SDR(py::module& m)
    {
        py::class_<SDR> py_SDR(m, "SDR");


        py_SDR.def(
            py::init<vector<UInt>>(),
            py::arg("inputDimensions")
        );

        py_SDR.def(
            py::init<SDR>(),
            py::arg("deepCopy")
        );

        // TODO: Deconstruct

        py_SDR.def_property_readonly("dimensions",
            [](const SDR &self) {
                return self.dimensions; });

        py_SDR.def_property_readonly("size",
            [](const SDR &self) {
                return self.size; });

        py_SDR.def("zero", &SDR::zero);


        // TODO: Reshape dense to correct dimensions!
        py_SDR.def_property("dense",
            [](SDR &self) {
                auto capsule = py::capsule(&self, [](void *self) {});
                return py::array(self.size, self.getDense().data(), capsule);
            },
            [](SDR &self, SDR_dense_t data) {
                // TODO: CHECK DATA SIZE & DIMS VALID!
                self.setDense( data );
            }
        );

        // getters setters for sparse, index

        py_SDR.def("getSum",      &SDR::getSum);
        py_SDR.def("getSparsity", &SDR::getSparsity);
        py_SDR.def("overlap",     &SDR::overlap);

        // randomize
        // addNoise
        // __repr__ & __str__
        // ==
        // !=
        // save/load
        //      * Save/load should be parent class inherited (serializable)

    }
}
