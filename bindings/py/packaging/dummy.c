#include <stdio.h>
#include <Python.h>

int main()
{
    return 0;
}

#if PY_MAJOR_VERSION >= 3
void PyInit_dummy() {}
#else
void initdummy() {}
#endif

