# Convenience wrappers. CMake (C/C++/CUDA libraries) and pip (Python package)
# are the real build systems; see docs/development.md.

BUILD ?= build
CMAKE_FLAGS ?=
PYTHON ?= python3

.PHONY: help build test cuda asan bench python python-test lint clean

help:
	@echo "make build        Configure and build the C/C++ libraries, tests and benchmark ($(BUILD)/)"
	@echo "make test         Build and run the C/C++ test suite (includes emulated CUDA parity tests)"
	@echo "make cuda         Build and test with the native CUDA backend (build-cuda/)"
	@echo "make asan         Build and test with AddressSanitizer + UBSan (build-asan/)"
	@echo "make bench        Run the C ABI microbenchmark"
	@echo "make python       Install the Python package in editable mode with test extras"
	@echo "make python-test  Run the Python test suite"
	@echo "make lint         Lint Python sources and check version metadata"
	@echo "make clean        Remove build trees and in-place extension modules"

build:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	cmake --build $(BUILD) --parallel

test: build
	ctest --test-dir $(BUILD) --output-on-failure

cuda:
	$(MAKE) test BUILD=build-cuda CMAKE_FLAGS="-DDBS_ENABLE_CUDA=ON $(CMAKE_FLAGS)"

asan:
	cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DDBS_ENABLE_SANITIZERS=ON $(CMAKE_FLAGS)
	cmake --build build-asan --parallel
	ctest --test-dir build-asan --output-on-failure

bench: build
	./$(BUILD)/dbs_bench

python:
	$(PYTHON) -m pip install --no-build-isolation -e ".[test]"

python-test:
	$(PYTHON) -m pytest python/tests

lint:
	ruff check python setup.py benchmarks examples experiments scripts
	ruff format --check python setup.py benchmarks examples experiments scripts
	$(PYTHON) scripts/check_version_metadata.py

clean:
	rm -rf build build-* dist python/*.egg-info python/beamgrad/_*.so python/beamgrad/_*.pyd
