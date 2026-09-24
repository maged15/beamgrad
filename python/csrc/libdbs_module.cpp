/* SPDX-License-Identifier: MIT
 *
 * beamgrad._libdbs is libdbs packaged as a Python extension file so the wheel
 * can ship it. It is loaded with ctypes (beamgrad/_ctypes.py); this module
 * object itself is empty. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef libdbs_module = {
    PyModuleDef_HEAD_INIT, "_libdbs", "libdbs C ABI (load with ctypes; see beamgrad._ctypes)", -1, NULL,
};

PyMODINIT_FUNC PyInit__libdbs(void) { return PyModule_Create(&libdbs_module); }
