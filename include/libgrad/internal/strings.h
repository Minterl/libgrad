#ifndef LG_FORMAT_H
#define LG_FORMAT_H

#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

typedef struct
lg_str8 {
    size_t    len;
    uint8_t  *p  lg_check_bounds(len);
} lg_str8;

// We use sizeof(str) - 1 to trim the null terminator
#define lg_str8_lit(str) ((lg_str8){ .len = sizeof(str) - 1, .p = (uint8_t*)(str) })

typedef struct
lg_writer {
    void *ctx;
    /// Must copy `str`, since it is not guaranteed to live any longer
    /// than the lifetime of the call to `Write`.
    ///
    /// Must return the number of bytes written.
    size_t (*write)(void *ctx, const lg_str8 str);
} lg_writer;

/// Behaves like libc `strcmp`, but returns 1 when `a.len` > `b.len`, and -1 in the opposite case.
int32_t 
lg_strcmp(const lg_str8 a, const lg_str8 b);

/// Copies from `src` to `dest` on a best-effor basis, meaning if `dest.len` < `src.len`, 
/// only `dest.len` bytes will ever be written, at a maximum.
///
/// Returns the number of bytes written.
size_t 
lg_strcpy(lg_str8 dest, const lg_str8 src);

void 
lg_copy_to_cstring(uint8_t *dst, const lg_str8 src);

#define lg_write(writer, string) (writer)->write((writer)->ctx, (string))

LG_StatusKind 
lg_printf(lg_writer *writer, const lg_str8 fmt, ...);

LG_StatusKind 
lg_vprintf(lg_writer *writer, const lg_str8 fmt, va_list ap);

void lg_write_itoa(lg_writer *writer, int64_t n);

#endif // LG_FORMAT_H
