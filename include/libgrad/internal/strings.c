#include <libgrad/internal/core.h>
#include <libgrad/internal/strings.h>
#include <libgrad/internal/fnv.h>
#include <libgrad/internal/debug.h>

#include <stdarg.h>
#include <stdint.h>

struct lg__fmt_fn_lut {
    uint32_t hash;
    void (*fn)(va_list ap, struct lg_writer *writer);
};

void LG__VFormatI64(va_list ap, struct lg_writer *writer) {
    int64_t arg = va_arg(ap, int64_t);
    LG__WriteItoa(writer, arg); 
}
void LG__VFormatString(va_list ap, struct lg_writer *writer) {
    struct lg_string s = va_arg(ap, struct lg_string);
    LG_Write(writer, s);
}
void LG__VFormatCString(va_list ap, struct lg_writer *writer) {
    uint8_t *s = va_arg(ap, uint8_t*);
    size_t len = 0;
    while (s[len] != '\0') {len++;};
    LG_Write(writer, ((struct lg_string){ .len = len, .p = s }));
}

#define LG__FMT_FN_LUT_LEN 3
static const struct lg__fmt_fn_lut LG__FMT_FN_LUT[LG__FMT_FN_LUT_LEN] = {
    {LG_HASH_LITERAL_16("i64"),   LG__VFormatI64},
    {LG_HASH_LITERAL_16("str"),   LG__VFormatString},
    {LG_HASH_LITERAL_16("cstr"),  LG__VFormatCString},
};

int32_t LG_Strcmp(const struct lg_string a, const struct lg_string b) {
    if (a.p == b.p && a.len == b.len) {
        return 0;
    }
    if (a.len != b.len) {
        return a.len > b.len ? 1 : -1;
    }

#   if defined(__has_builtin) && __has_builtin(__builtin_memcmp)
        return __builtin_memcmp(a.p, b.p, a.len);
#   else
        for (size_t i = 0; i < a.len; i++) {
            if (a.p[i] == b.p[i]) {
                continue;
            } else {
                return (int32_t)a.p[i] - (int32_t)b.p[i];
            }
        }
        return 0;
#   endif // defined(__has_builtin) && __has_builtin(__builtin_memcmp)
}

size_t LG_Strcpy(struct lg_string dest, const struct lg_string src) {
    size_t i = 0;
    for (; i < dest.len && i < src.len; i++) {
        dest.p[i] = src.p[i];
    }
    return i;
}

void LG_CopyToCString(uint8_t *dest, const struct lg_string src) {
    LG__Assert(dest != NULL);
    LG__Assert(src.p != NULL);

    size_t i = 0;
    for (; i < src.len; i++) {
        dest[i] = src.p[i];
    }
    dest[i] = '\0';
}

void LG__WriteItoa(struct lg_writer *writer, int64_t n) {
    _Static_assert(INT64_MAX == 9223372036854775807, "");
    //          ... which is -- 1234567890123456789 -- 19 digits long
    // +1 for the sign character.
    // `lg_string` does not need a null terminator
    uint8_t buf[20] = {0};
    size_t len = 0;

    uint64_t abs_n = (n < 0) ? (uint64_t)-(n + 1) + 1 : (uint64_t)n;
    do {
        buf[len] = '0' + (abs_n % 10);
        len++;
        abs_n /= 10;
    } while (abs_n > 0);

    if (n < 0) {
        buf[len] = '-';
        len++;
    }

    for (size_t i = 0; i < len / 2; i++) {
        const size_t i_left = i;
        const size_t i_right = len - 1 - i;
        const uint8_t temp = buf[i_left];
        buf[i_left] = buf[i_right];
        buf[i_right] = temp;
    }

    LG_Write(writer, ((struct lg_string){ .len = len, .p = buf }));
}

enum lg_status LG_Printf(struct lg_writer *writer, const struct lg_string fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    enum lg_status status = LG_VPrintf(writer, fmt, ap);
    va_end(ap);
    return status;
}

enum lg_status LG_VPrintf(struct lg_writer *writer, const struct lg_string fmt, va_list ap) {
    for (size_t i = 0; i < fmt.len; i++) {
        if (
            fmt.p[i] != '%' ||
            (i + 1) >= fmt.len ||
            fmt.p[i + 1] != '{'
        ) {
            LG_Write(writer, ((struct lg_string){ .len = 1, .p = fmt.p + i }));
            continue;
        }

        struct lg_string fmtspec;
        {
            LG__Assert(fmt.p[i] == '%');
            LG__Assert(fmt.p[i + 1] == '{');

            const size_t fmtspec_begin = i + 2;
            size_t scan = fmtspec_begin;
            while (scan < fmt.len && fmt.p[scan] != '}') {
                scan++;
                // unterminated format specifier
                if (scan == fmt.len - 1) {
                    return LG_STATUS_INVALID_ARGUMENT;
                }
            }      
            const size_t fmtspec_end = scan;
            i = fmtspec_end;

            fmtspec = (struct lg_string){
                .len = fmtspec_end - fmtspec_begin,
                .p = fmt.p + fmtspec_begin,
            };
            if (fmtspec.len == 0) {
                return LG_STATUS_INVALID_ARGUMENT;
            }
            LG__Assert((fmtspec.p + fmtspec.len) < (fmt.p + fmt.len));
        }

        uint32_t hash = LG_HASH_16(fmtspec.p, (fmtspec.len < 16 ? fmtspec.len : 16));
        bool found = false;
        for (size_t i = 0; i < LG__FMT_FN_LUT_LEN; i++) {
            if (LG__FMT_FN_LUT[i].hash == hash) {
                LG__FMT_FN_LUT[i].fn(ap, writer);
                found = true;
                break;
            }
        }
        if (!found) {
            return LG_STATUS_INVALID_ARGUMENT;
        }

        LG__Assert(i < fmt.len);
    }

    return LG_STATUS_OK;
}
