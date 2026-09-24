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


def test_c_errors_carry_the_library_message():
    lib = _ctypes.load_library()
    bad = _ctypes.options_to_c(BeamOptions(beam_size=2))
    bad.beam_size = -3
    handle = ctypes.c_void_p()
    # DBS_ERROR_INVALID_ARGUMENT (-1) becomes ValueError, other failures RuntimeError.
    with pytest.raises(ValueError, match="beam_size"):
        _ctypes.check(lib, None, lib.dbs_create_ex(bad, ctypes.byref(handle)))


def test_validate_inputs_is_passed_to_the_library():
    assert _ctypes.options_to_c(BeamOptions(beam_size=2)).validate_inputs == 1
    assert _ctypes.options_to_c(BeamOptions(beam_size=2, validate_inputs=False)).validate_inputs == 0
    assert _ctypes.constraints_to_c(BeamOptions(beam_size=2), None) is None
