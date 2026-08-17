#include <libgrad/internal/base.h>

int32_t
lg_memcmp_(uint8_t *a, uint8_t *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) {
            return (int32_t)a[i] - (int32_t)b[i];
        }
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// string implementation stuff
///
////////////////////////////////////////////////////////////////////////////////

void 
lg_vformat_i64(va_list ap, LG_Writer *writer) {
    int64_t arg = va_arg(ap, int64_t);
    lg_write_itoa(writer, arg); 
}
void 
lg_vformat_string(va_list ap, LG_Writer *writer) {
    lg_str8 s = va_arg(ap, lg_str8);
    lg_write(writer, s);
}
void 
lg_vformat_cstring(va_list ap, LG_Writer *writer) {
    uint8_t *s = va_arg(ap, uint8_t*);
    size_t len = 0;
    while (s[len] != '\0') {len++;};
    lg_write(writer, ((lg_str8){ .len = len, .p = s }));
}
void lg_vformat_status_kind(va_list ap, LG_Writer *writer) {
    LG_StatusKind status = va_arg(ap, LG_StatusKind);
    uint8_t *str = (uint8_t*)lg_status_kind_as_cstring(status);
    size_t len = 0;
    while (str[len] != '\0') {len++;};
    lg_write(writer, ((lg_str8){ .len = len, .p = str }));
}

#define LG_FMT_FN_LUT_LEN 4
static const struct {
    uint32_t hash;
    void (*fn)(va_list ap, LG_Writer *writer);
} LG_FMT_FN_LUT[LG_FMT_FN_LUT_LEN] = {
    {lg_hash_lit_16("i64"),     lg_vformat_i64},
    {lg_hash_lit_16("str"),     lg_vformat_string},
    {lg_hash_lit_16("cstr"),    lg_vformat_cstring},
    {lg_hash_lit_16("status"),  lg_vformat_status_kind},
};

int32_t 
lg_strcmp(const lg_str8 a, const lg_str8 b) {
    if (a.p == b.p && a.len == b.len) {
        return 0;
    }
    const size_t len = a.len > b.len ? b.len : a.len;
    return lg_memcmp(a.p, b.p, len);
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
lg_write_itoa(LG_Writer *writer, int64_t n) {
    lg_static_assert(INT64_MAX == 9223372036854775807);
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
lg_printf(LG_Writer *writer, const lg_str8 fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    LG_StatusKind status = lg_vprintf(writer, fmt, ap);
    va_end(ap);
    return status;
}

LG_StatusKind 
lg_vprintf(LG_Writer *writer, const lg_str8 fmt, va_list ap) {
    LG_StatusKind status = LG_StatusKind_OK;

    for (size_t i = 0; i < fmt.len; i++) {
        if (
            fmt.p[i] != '%' ||
            (i + 1) >= fmt.len ||
            fmt.p[i + 1] != '{'
        ) {
            lg_write(writer, ((lg_str8){ .len = 1, .p = fmt.p + i }));
            continue;
        }


        ////////////////////////////////////////////////// 
        // ~~ Parse the format specifier ~~

        lg_str8 fmtspec;
        {
            lg_assert(fmt.p[i] == '%');
            lg_assert(fmt.p[i + 1] == '{');
            lg_assert(fmt.len > i + 2);

            size_t fmtspec_begin = i + 2;
            size_t fmtspec_end = fmtspec_begin;
            {
                while (fmt.p[fmtspec_end] != '}') {
                    // unterminated format specifier
                    if (fmtspec_end >= fmt.len - 1) {
                        status = LG_StatusKind_InvalidArgument;
                        goto out;
                    }
                    fmtspec_end++;
                }      
                i = fmtspec_end; // i will be incremeted at the bottom of the loop
            }

            fmtspec = (lg_str8){
                .len = fmtspec_end - fmtspec_begin,
                .p = fmt.p + fmtspec_begin,
            };
            if (fmtspec.len == 0) {
                status = LG_StatusKind_InvalidArgument;
                goto out;
            }

            lg_assert((fmtspec.p + fmtspec.len) < (fmt.p + fmt.len));
        }


        ////////////////////////////////////////////////// 
        // ~~ Format specifier LUT lookup ~~
        {
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
                status = LG_StatusKind_InvalidArgument;
                goto out;
            }
        }

        lg_assert(i < fmt.len);
    }

out:
    if (status != LG_StatusKind_OK) {
        lg_write(writer, lg_str8_lit("(error)"));
    }
    return LG_StatusKind_OK;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// guarded debug utilities
///
////////////////////////////////////////////////////////////////////////////////

// Use a guard here b/c this block expects libc,
// which may not be available.
#ifdef LG_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void 
lg_dbgf_(const char *file, int line, const char* fmt, ...) {
    fprintf(stderr, "\033[32m[DEBUG]\033[0m (%s:%d) -- ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void 
lg_assert_(const char *file, int line, bool cond, const char *cond_str) {
    if (!cond) {
        fprintf(stderr, "\x1b[31m[ASSERTION FAILED]\x1b[0m (%s) at %s:%d\n", cond_str, file, line);
        abort();
    }
}

#endif // LG_DEBUG


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// 
/// allocator implemementations
///
////////////////////////////////////////////////////////////////////////////////

uint8_t*
lg_alloc_zero(LG_Allocator *alloc, size_t size_bytes) {
    uint8_t *ptr = alloc->alloc(alloc->ctx, size_bytes);
    if (ptr == NULL) {
        return ptr;
    }

    if (!(alloc->flags & LG_AllocatorFlag_AssumeZeroed)) {
        lg_memzero(ptr, size_bytes);
    }

    return ptr;
}

void 
lg_free(LG_Allocator *alloc, void *ptr) {
    if (alloc->free) {
        alloc->free(alloc->ctx, ptr);
    }
}

LG_StatusKind 
lg_alloc_contiguous_blocks(
    LG_Allocator *alloc,
    uint8_t **out_ptrs,
    size_t *lg_nullable out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        size += lg_align_up(sizes[i], align);
    }

    uint8_t *ptr = lg_alloc_zero(alloc, size);
    if (ptr == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < n; i++) {
        out_ptrs[i] = ptr + current_offset;
        current_offset += lg_align_up(sizes[i], align);
    }

    return LG_StatusKind_OK;
}


lg_force_inline void
lg_slab_unlink(LG_Slab *slab) {
    if (slab->prev != NULL) {
        slab->prev->next = slab->next;
    }            
    if (slab->next != NULL) {
        slab->next->prev = slab->prev;
    }            
    slab->prev = NULL;
    slab->next = NULL;
}

lg_force_inline void
lg_slab_append(LG_Slab *slab, LG_Slab *onto) {
    while (true) {
        lg_assert(onto != slab);
        if (onto->next == NULL) {
            break;
        }
        onto = onto->next;
    }

    onto->next = slab;
    slab->prev = onto;
}

lg_force_inline void
lg_slab_free_from(LG_Slab *slab, LG_Allocator *alloc) {
    LG_Slab *next = slab;
    while (next != NULL) {
        LG_Slab *current = next;
        next = current->next;
        lg_free(alloc, current);
    }
}

lg_force_inline LG_Slab*
lg_slab_find_first(LG_Slab *slab) {
    bool is_first_iteration = true;
    LG_Slab *first = slab;
    while (true) {
        lg_assert(is_first_iteration == true || first != slab);
        is_first_iteration = false;
        if (first->prev == NULL) {
            break;
        }
        first = first->prev;
    }
    return first;
}


void
lg_arena_init(LG_Arena *arena, LG_Allocator *host) {
    lg_memzero(arena, sizeof(LG_Arena));
    arena->host = host;
}

uint8_t*
lg_arena_alloc(LG_Arena *arena, size_t unaligned_size_bytes, size_t align) {
    const size_t size_bytes = lg_align_up(unaligned_size_bytes, align);


    ////////////////////////////////////////
    // ~~ Plan A: use the current slab ~~

    if (lg_likely(
        arena->current_slab != NULL &&
        arena->current_offset + size_bytes <= arena->current_slab->cap
    )) {
        lg_memzero(arena->current_slab->buf + arena->current_offset, size_bytes);

        const size_t prev_offset = arena->current_offset;
        arena->current_offset += size_bytes;

        return arena->current_slab->buf + prev_offset;
    }


    //////////////////////////////////////////////////
    // ~~ Plan B: first-fit an existing free slab ~~

    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        goto plan_c;
    }

    lg_assert(arena->recycled_slabs_head == NULL || arena->recycled_slabs_head->prev == NULL);

    LG_Slab *to_reuse = arena->recycled_slabs_head;
    while (to_reuse != NULL) {
        if (to_reuse->cap >= size_bytes) {
            lg_slab_unlink(to_reuse);

            if (arena->current_slab != NULL) {
                arena->current_slab->next = to_reuse;
                to_reuse->prev = arena->current_slab;
            }

            arena->current_slab = to_reuse;
            arena->current_offset = size_bytes;

            lg_memzero(to_reuse->buf, size_bytes);

            return to_reuse->buf;
        }

        to_reuse = to_reuse->next;
    }


    ////////////////////////////////////////
    // ~~ Plan C: allocate a new slab ~~

plan_c:;

    const size_t buf_size = size_bytes > arena->host->default_slab_size_bytes ?
        size_bytes :
        arena->host->default_slab_size_bytes;
    const size_t total_size = sizeof(LG_Slab) + buf_size;

    LG_Slab *next = arena->host->alloc(arena->host->ctx, total_size);
    if (next == NULL) {
        return NULL;
    }

    lg_memzero(next, sizeof(LG_Slab));
    next->cap = buf_size;

    if (arena->current_slab != NULL) {
        lg_assert(arena->current_slab->next == NULL);
        arena->current_slab->next = next;
        next->prev = arena->current_slab;
    }

    arena->current_slab = next;
    arena->current_offset = size_bytes;

    lg_memzero(next->buf, size_bytes);

    return next->buf;
}

LG_Scope
lg_push_scope(LG_Arena *arena) {
    return (LG_Scope){
        .offset = arena->current_offset,
        .slab = arena->current_slab,
    };
}

void
lg_pop_scope(LG_Arena *arena, LG_Scope scope) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        if (scope.slab == arena->current_slab) {
            arena->current_slab = NULL;
        }

        lg_slab_free_from(scope.slab, arena->host);

        return;
    }

    LG_Slab *to_recycle = scope.offset == 0 ?
        scope.slab :
        scope.slab->next;

    if (to_recycle == NULL) {
        return;
    }
    if (to_recycle == arena->current_slab) {
        arena->current_slab = NULL;
    }
    if (to_recycle->prev != NULL) {
        to_recycle->prev->next = NULL;
    }

    lg_slab_append(to_recycle, arena->recycled_slabs_head);

    arena->current_offset = scope.offset;
}

void
lg_arena_free_recycled(LG_Arena *arena) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        lg_assert(arena->recycled_slabs_head == NULL);
        return;
    }
    if (arena->recycled_slabs_head == NULL) {
        return;
    }

    lg_assert(arena->recycled_slabs_head->prev == NULL);
    lg_slab_free_from(arena->recycled_slabs_head, arena->host);
    arena->recycled_slabs_head = NULL;
}

void
lg_arena_recycle_all(LG_Arena *arena) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        return;
    }

    LG_Slab *first_active = lg_slab_find_first(arena->current_slab);
    lg_slab_append(first_active, arena->recycled_slabs_head);
    arena->current_slab = NULL;
}

void
lg_arena_free_all(LG_Arena *arena) {
    LG_Slab *first_active = lg_slab_find_first(arena->current_slab);
    lg_slab_free_from(first_active, arena->host);
    lg_arena_free_recycled(arena);
    arena->current_offset = 0;
    arena->current_slab = NULL;
}
