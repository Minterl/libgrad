#ifndef LG_TESTING_H_
#define LG_TESTING_H_

#include <stdbool.h>

#define assert(cond, ...) do { \
    if (!(cond)) {\
        __assert_fail(#cond, __FILE__, __LINE__, __VA_ARGS__); \
        return; \
    } \
} while(0)

void __assert_fail(const char *cond_str, const char *file, int line, const char *fmt, ...);

#endif // LG_TESTING_H_

#ifdef LG_TESTING_IMPLEMENTATION
#undef LG_TESTING_IMPLEMENTATION

#include <stdio.h>
#include <stdarg.h>

void __assert_fail(const char *cond_str, const char *file, int line, const char *fmt, ...) {
    fprintf(stderr, "[ASSERTION FAILED] (%s) at %s:%d -- ", cond_str, file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#endif // LG_TESTING_IMPLEMENTATION
