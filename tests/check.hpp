// SPDX-License-Identifier: MIT
//
// Always-on test assertion. Unlike assert(), CHECK is not compiled out by
// NDEBUG, so the test suite verifies the same things in Release and Debug.
#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                             \
        }                                                                             \
    } while (0)
