#ifndef __PYX_HAVE__create_example
#define __PYX_HAVE__create_example


#ifndef __PYX_HAVE_API__create_example

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

#include <capnp/dynamic.h>

__PYX_EXTERN_C DL_IMPORT(PyObject) *createReader( ::capnp::DynamicStruct::Reader, PyObject *);
__PYX_EXTERN_C DL_IMPORT(PyObject) *createBuilder( ::capnp::DynamicStruct::Builder, PyObject *);

#endif /* !__PYX_HAVE_API__create_example */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcreate_example(void);
#else
PyMODINIT_FUNC PyInit_create_example(void);
#endif

#endif /* !__PYX_HAVE__create_example */
