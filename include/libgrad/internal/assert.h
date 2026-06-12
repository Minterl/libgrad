#ifndef LG_INTERNAL_ASSERT_H_
#define LG_INTERNAL_ASSERT_H_

#include <stdbool.h>
#include <stdlib.h>

#ifdef LG_ASSERTIONS
#define lg_assert(msg, cond) (__lg_assert(msg, cond, __FILE__, __LINE__))
#else
#define lg_assert(msg, cond) /* nothing */
#endif // LG_ASSERTIONS

void __lg_assert(const char *msg, bool cond, const char *file, int line);

#endif // LG_INTERNAL_ASSERT_H_

#if defined(LG_INTERNAL_ASSERT_IMPLEMENTATION) && defined(LG_ASSERTIONS)
#undef LG_INTERNAL_ASSERT_IMPLEMENTATION

#include <stdio.h>

void __lg_assert(const char *msg, bool cond, const char *file, int line) {
    if (!cond) {
        fprintf(stderr, "Assertion failed at %s:%d -- %s\n", file, line, msg);
        exit(1);
    }
}

#endif // LG_INTERNAL_ASSERT_IMPLEMENTATION
