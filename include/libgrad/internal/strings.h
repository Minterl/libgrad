#ifndef LG_FORMAT_H
#define LG_FORMAT_H

#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

struct lg_string {
    size_t    len;
    uint8_t  *p  LG_CHECK_BOUNDS(len);
};

// We use sizeof(str) - 1 to trim the null terminator
#define LG_STRING_LITERAL(str) ((struct lg_string){ .len = sizeof(str) - 1, .p = (uint8_t*)(str) })

struct lg_writer {
    void *ctx;
    /// Must copy `str`, since it is not guaranteed to live any longer
    /// than the lifetime of the call to `Write`.
    ///
    /// Must return the number of bytes written.
    size_t (*Write)(void *ctx, const struct lg_string str);
};

/// Behaves like libc `strcmp`, but returns 1 when `a.len` > `b.len`, and -1 in the opposite case.
int32_t LG_Strcmp(const struct lg_string a, const struct lg_string b);

/// Copies from `src` to `dest` on a best-effor basis, meaning if `dest.len` < `src.len`, 
/// only `dest.len` bytes will ever be written, at a maximum.
///
/// Returns the number of bytes written.
size_t LG_Strcpy(struct lg_string dest, const struct lg_string src);

void LG_CopyToCString(uint8_t *dst, const struct lg_string src);

#define LG_Write(writer, string) (writer)->Write((writer)->ctx, (string))
enum lg_status LG_Printf(struct lg_writer *writer, const struct lg_string fmt, ...);
enum lg_status LG_VPrintf(struct lg_writer *writer, const struct lg_string fmt, va_list ap);

void LG__WriteItoa(struct lg_writer *writer, int64_t n);

#endif // LG_FORMAT_H
