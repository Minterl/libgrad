#include <libgrad/internal/core.h>
#include <libgrad/internal/strings.h>
#include <libgrad/internal/fnv.h>
#include <libgrad/internal/debug.h>

#include <stdarg.h>
#include <stdint.h>

typedef struct
LG_FmtFnLut {
    uint32_t hash;
    void (*fn)(va_list ap, lg_writer *writer);
} LG_FmtFnLut;

void 
lg_vformat_i64(va_list ap, lg_writer *writer) {
    int64_t arg = va_arg(ap, int64_t);
    lg_write_itoa(writer, arg); 
}
void 
lg_vformat_string(va_list ap, lg_writer *writer) {
    lg_str8 s = va_arg(ap, lg_str8);
    lg_write(writer, s);
}
void 
lg_vformat_cstring(va_list ap, lg_writer *writer) {
    uint8_t *s = va_arg(ap, uint8_t*);
    size_t len = 0;
    while (s[len] != '\0') {len++;};
    lg_write(writer, ((lg_str8){ .len = len, .p = s }));
}

#define LG_FMT_FN_LUT_LEN 3
static const LG_FmtFnLut LG_FMT_FN_LUT[LG_FMT_FN_LUT_LEN] = {
    {lg_hash_lit_16("i64"),   lg_vformat_i64},
    {lg_hash_lit_16("str"),   lg_vformat_string},
    {lg_hash_lit_16("cstr"),  lg_vformat_cstring},
};

int32_t 
lg_strcmp(const lg_str8 a, const lg_str8 b) {
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

size_t 
lg_strcpy(lg_str8 dest, const lg_str8 src) {
    size_t i = 0;
    for (; i < dest.len && i < src.len; i++) {
        dest.p[i] = src.p[i];
    }
    return i;
}

void 
lg_copy_to_cstring(uint8_t *dest, const lg_str8 src) {
    lg_assert(dest != NULL);
    lg_assert(src.p != NULL);

    size_t i = 0;
    for (; i < src.len; i++) {
        dest[i] = src.p[i];
    }
    dest[i] = '\0';
}

void 
lg_write_itoa(lg_writer *writer, int64_t n) {
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

    lg_write(writer, ((lg_str8){ .len = len, .p = buf }));
}

LG_StatusKind 
lg_printf(lg_writer *writer, const lg_str8 fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    LG_StatusKind status = lg_vprintf(writer, fmt, ap);
    va_end(ap);
    return status;
}

LG_StatusKind 
lg_vprintf(lg_writer *writer, const lg_str8 fmt, va_list ap) {
    for (size_t i = 0; i < fmt.len; i++) {
        if (
            fmt.p[i] != '%' ||
            (i + 1) >= fmt.len ||
            fmt.p[i + 1] != '{'
        ) {
            lg_write(writer, ((lg_str8){ .len = 1, .p = fmt.p + i }));
            continue;
        }

        lg_str8 fmtspec;
        {
            lg_assert(fmt.p[i] == '%');
            lg_assert(fmt.p[i + 1] == '{');

            const size_t fmtspec_begin = i + 2;
            size_t scan = fmtspec_begin;
            while (scan < fmt.len && fmt.p[scan] != '}') {
                scan++;
                // unterminated format specifier
                if (scan == fmt.len - 1) {
                    return LG_StatusKind_InvalidArgument;
                }
            }      
            const size_t fmtspec_end = scan;
            i = fmtspec_end;

            fmtspec = (lg_str8){
                .len = fmtspec_end - fmtspec_begin,
                .p = fmt.p + fmtspec_begin,
            };
            if (fmtspec.len == 0) {
                return LG_StatusKind_InvalidArgument;
            }
            lg_assert((fmtspec.p + fmtspec.len) < (fmt.p + fmt.len));
        }

        uint32_t hash = lg_hash_16(fmtspec.p, (fmtspec.len < 16 ? fmtspec.len : 16));
        bool found = false;
        for (size_t i = 0; i < LG_FMT_FN_LUT_LEN; i++) {
            if (LG_FMT_FN_LUT[i].hash == hash) {
                LG_FMT_FN_LUT[i].fn(ap, writer);
                found = true;
                break;
            }
        }
        if (!found) {
            return LG_StatusKind_InvalidArgument;
        }

        lg_assert(i < fmt.len);
    }

    return LG_StatusKind_OK;
}
