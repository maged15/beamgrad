# SPDX-License-Identifier: MIT
"""The bundled libdbs and its ctypes bindings."""

import ctypes

import pytest

import beamgrad
from beamgrad import BeamOptions, _ctypes


def test_options_struct_matches_c_layout():
    assert ctypes.sizeof(_ctypes.DBSOptionsC) == _ctypes.OPTIONS_SIZE
    for name, offset in _ctypes.OPTIONS_LAYOUT.items():
        assert getattr(_ctypes.DBSOptionsC, name).offset == offset


def test_options_conversion():
    c = _ctypes.options_to_c(BeamOptions(beam_size=5, eos_token=3, min_length=2, length_penalty_alpha=0.5))
    assert (c.beam_size, c.eos_token, c.min_length) == (5, 3, 2)
    assert c.length_penalty_alpha == pytest.approx(0.5)
    assert c.max_dense_gradient_elements == 0  # zero selects the library default


def test_bundled_library_loads():
    lib = _ctypes.load_library()
    assert lib.dbs_abi_version() == _ctypes.ABI_VERSION
    assert lib.dbs_version_string().decode() == beamgrad.__version__


def test_library_errors_are_actionable(tmp_path):
    missing = str(tmp_path / "libdbs-missing.so")
    with pytest.raises(RuntimeError, match="Unable to load libdbs"):
        _ctypes.load_library(missing)


def test_c_errors_surface_as_runtime_errors():
    lib = _ctypes.load_library()
    bad = _ctypes.options_to_c(BeamOptions(beam_size=2))
    bad.beam_size = -3
    handle = ctypes.c_void_p()
    with pytest.raises(RuntimeError, match="beam_size"):
        _ctypes.check(lib, None, lib.dbs_create_ex(bad, ctypes.byref(handle)))
