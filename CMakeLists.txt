cmake_minimum_required(VERSION 3.20)

project(beamgrad
  VERSION 2.0.0
  DESCRIPTION "Differentiable beam search: C ABI, SIMD CPU kernels, and native CUDA"
  HOMEPAGE_URL "https://github.com/maged15/beamgrad"
  LANGUAGES C CXX)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}" DBS_TOP_LEVEL)

option(DBS_BUILD_SHARED "Build shared libraries (static when OFF)" ON)
option(DBS_BUILD_TESTS "Build the test suite" ${DBS_TOP_LEVEL})
option(DBS_BUILD_BENCHMARKS "Build the C++ benchmark" ${DBS_TOP_LEVEL})
option(DBS_BUILD_FUZZER "Build the libFuzzer harness (Clang only)" OFF)
option(DBS_ENABLE_CUDA "Build the native CUDA backend (otherwise a stub is built)" OFF)
option(DBS_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(DBS_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(DBS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# The C ABI version is the shared-library SOVERSION; it changes only on
# binary-incompatible changes (see docs/c-api.md).
set(DBS_ABI_VERSION 10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

if(DBS_ENABLE_SANITIZERS AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()
if(DBS_ENABLE_TSAN AND NOT MSVC)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# Floating-point contraction (fused multiply-add) would let compilers round
# differently on different targets; results must not depend on the target.
function(dbs_set_fp_model target)
  if(NOT MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(dbs_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(DBS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic>)
    if(DBS_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
    endif()
  endif()
endfunction()

if(DBS_BUILD_SHARED)
  set(DBS_LIBRARY_TYPE SHARED)
else()
  set(DBS_LIBRARY_TYPE STATIC)
endif()

# ---------------------------------------------------------------------------
# libdbs: CPU decoder and C ABI
# ---------------------------------------------------------------------------

set(DBS_CORE_SOURCES
  src/c_api.cpp
  src/cpu_features.cpp
  src/decoder.cpp
  src/kernels_scalar.cpp
  src/kernels_x86.cpp)

# Object library so the internal tests can link the implementation directly.
add_library(dbs_core_objects OBJECT ${DBS_CORE_SOURCES})
set_target_properties(dbs_core_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dbs_core_objects PRIVATE include src)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs_core_objects PRIVATE DBS_BUILD_SHARED DBS_COMPILING_LIBRARY)
else()
  target_compile_definitions(dbs_core_objects PRIVATE DBS_STATIC)
endif()
dbs_set_warnings(dbs_core_objects)
dbs_set_fp_model(dbs_core_objects)

add_library(dbs ${DBS_LIBRARY_TYPE} $<TARGET_OBJECTS:dbs_core_objects>)
add_library(beamgrad::dbs ALIAS dbs)
target_include_directories(dbs PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
find_package(Threads REQUIRED)
target_link_libraries(dbs PRIVATE Threads::Threads)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs INTERFACE DBS_BUILD_SHARED)
  set_target_properties(dbs PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${DBS_ABI_VERSION})
  if(UNIX AND NOT APPLE)
    target_link_options(dbs PRIVATE "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/dbs.map")
  endif()
else()
  target_compile_definitions(dbs INTERFACE DBS_STATIC)
endif()

# ---------------------------------------------------------------------------
# libdbs_cuda: native CUDA backend, or a stub that reports "unavailable"
# ---------------------------------------------------------------------------

if(DBS_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
  endif()
  enable_language(CUDA)
  add_library(dbs_cuda ${DBS_LIBRARY_TYPE} cuda/dbs_cuda.cu)
  set_target_properties(dbs_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON)
else()
  add_library(dbs_cuda ${DBS_LIBRARY_TYPE} cuda/dbs_cuda_stub.cpp)
  dbs_set_warnings(dbs_cuda)
endif()
add_library(beamgrad::dbs_cuda ALIAS dbs_cuda)
target_include_directories(dbs_cuda
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE src)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs_cuda PRIVATE DBS_BUILD_SHARED DBS_COMPILING_LIBRARY INTERFACE DBS_BUILD_SHARED)
  set_target_properties(dbs_cuda PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${DBS_ABI_VERSION})
else()
  target_compile_definitions(dbs_cuda PUBLIC DBS_STATIC)
endif()

# ---------------------------------------------------------------------------
# Tests, benchmark, fuzzer
# ---------------------------------------------------------------------------

if(DBS_BUILD_TESTS)
  enable_testing()

  add_executable(dbs_tests tests/dbs_tests.cpp)
  target_link_libraries(dbs_tests PRIVATE dbs)
  dbs_set_warnings(dbs_tests)
  add_test(NAME dbs_tests COMMAND dbs_tests)

  add_executable(dbs_internal_tests tests/internal_tests.cpp $<TARGET_OBJECTS:dbs_core_objects>)
  target_include_directories(dbs_internal_tests PRIVATE include src tests)
  target_compile_definitions(dbs_internal_tests PRIVATE DBS_STATIC)
  target_link_libraries(dbs_internal_tests PRIVATE Threads::Threads)
  dbs_set_warnings(dbs_internal_tests)
  dbs_set_fp_model(dbs_internal_tests)
  add_test(NAME dbs_internal_tests COMMAND dbs_internal_tests)

  # Both public headers must be valid C, and both libraries must export them.
  add_executable(dbs_c_api_test tests/c_api_test.c)
  target_link_libraries(dbs_c_api_test PRIVATE dbs dbs_cuda)
  dbs_set_warnings(dbs_c_api_test)
  add_test(NAME dbs_c_api_test COMMAND dbs_c_api_test)

  # The documented C example must keep compiling and running.
  add_executable(dbs_example_c_api examples/c_api.c)
  target_link_libraries(dbs_example_c_api PRIVATE dbs)
  dbs_set_warnings(dbs_example_c_api)
  add_test(NAME dbs_example_c_api COMMAND dbs_example_c_api)

  if(DBS_ENABLE_SANITIZERS)
    # A UBSan-instrumented libdbs needs the sanitizer's C++ runtime, which only
    # the C++ driver links (Clang); plain C consumers are covered by normal builds.
    set_target_properties(dbs_c_api_test dbs_example_c_api PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  # The CUDA backend source, executed on the CPU and checked against libdbs.
  # Needs POSIX ucontext; Linux is enough to verify the kernel logic.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(dbs_cuda_emulation_tests tests/cuda_emulation_tests.cpp)
    target_include_directories(dbs_cuda_emulation_tests PRIVATE include src tests)
    target_link_libraries(dbs_cuda_emulation_tests PRIVATE dbs)
    dbs_set_fp_model(dbs_cuda_emulation_tests)
    add_test(NAME dbs_cuda_emulation_tests COMMAND dbs_cuda_emulation_tests)
    if(DBS_ENABLE_SANITIZERS)
      # ASan cannot follow ucontext stack switches.
      set_tests_properties(dbs_cuda_emulation_tests PROPERTIES
        ENVIRONMENT "ASAN_OPTIONS=detect_stack_use_after_return=0:detect_leaks=1")
    endif()
  endif()

  # The native CUDA backend on a real GPU against libdbs, bit for bit. Skipped
  # (exit code 77) when no device is present.
  if(DBS_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(dbs_cuda_device_tests tests/cuda_device_tests.cpp)
    target_include_directories(dbs_cuda_device_tests PRIVATE tests)
    target_link_libraries(dbs_cuda_device_tests PRIVATE dbs dbs_cuda CUDA::cudart)
    dbs_set_warnings(dbs_cuda_device_tests)
    add_test(NAME dbs_cuda_device_tests COMMAND dbs_cuda_device_tests)
    set_tests_properties(dbs_cuda_device_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  if(UNIX AND NOT APPLE AND DBS_BUILD_SHARED)
    add_test(NAME dbs_abi_symbols
      COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/check_abi.sh $<TARGET_FILE:dbs>)
  endif()
endif()

if(DBS_BUILD_BENCHMARKS)
  add_executable(dbs_bench benchmarks/dbs_bench.cpp)
  target_link_libraries(dbs_bench PRIVATE dbs)
endif()

if(DBS_BUILD_FUZZER)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "DBS_BUILD_FUZZER requires Clang (libFuzzer)")
  endif()
  # The harness with its own copy of the library, all instrumented, so that
  # libFuzzer is guided by coverage of the code under test and not only of the
  # harness. The shared libdbs stays uninstrumented.
  add_executable(dbs_fuzz tests/fuzz_dbs.cpp ${DBS_CORE_SOURCES})
  target_include_directories(dbs_fuzz PRIVATE include src)
  target_compile_definitions(dbs_fuzz PRIVATE DBS_STATIC)
  dbs_set_warnings(dbs_fuzz)
  dbs_set_fp_model(dbs_fuzz)
  target_link_libraries(dbs_fuzz PRIVATE Threads::Threads)
  target_compile_options(dbs_fuzz PRIVATE -fsanitize=fuzzer)
  target_link_options(dbs_fuzz PRIVATE -fsanitize=fuzzer)
endif()

# ---------------------------------------------------------------------------
# Install: headers, libraries, CMake package (find_package(beamgrad)), pkg-config
# ---------------------------------------------------------------------------

install(TARGETS dbs dbs_cuda EXPORT beamgradTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/dbs.h include/dbs_cuda.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT beamgradTargets
  FILE beamgradTargets.cmake
  NAMESPACE beamgrad::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/beamgrad)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/beamgradConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/beamgradConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/beamgradConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/beamgrad)
install(FILES
  "${PROJECT_BINARY_DIR}/beamgradConfig.cmake"
  "${PROJECT_BINARY_DIR}/beamgradConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/beamgrad)

configure_file(pkgconfig/dbs.pc.in dbs.pc @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/dbs.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
