#ifndef ARENA_H_
#define ARENA_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include <libgrad/libgrad.h>

typedef struct com_arena {
    size_t cap;
    size_t offset;
    uint8_t *buf;
} com_arena;

static inline uint8_t com_arena_init(com_arena *a, size_t cap) {
    a->buf = malloc(cap);
    if (a->buf == NULL) {
        return -1;
    }
    a->cap = cap;
    a->offset = 0;
    return 0;
}

static inline void com_arena_destroy(com_arena *a) {
    free(a->buf);
}

static inline uint8_t *com_arena_alloc(com_arena *a, size_t size_bytes) {
    if (a->offset + size_bytes > a->cap) {
        return NULL;
    }
    uint8_t *ret = a->buf + a->offset;
    a->offset += size_bytes;
    return ret;
}

static inline void com_arena_reset(com_arena *a) {
    a->offset = 0; 
}

static inline void *__com_arena_as_lg_allocator_alloc(void *ctx, size_t size_bytes) {
    com_arena *const a = ctx;
    return (void*)com_arena_alloc(a, size_bytes);
}

static inline void __com_arena_as_lg_allocator_free(void *ctx, void *ptr) {
    (void)ctx;
    (void)ptr;
}

static inline lg_allocator com_arena_as_lg_allocator(com_arena *a) {
    return (lg_allocator) {
        .ctx = a,
        .alloc = __com_arena_as_lg_allocator_alloc,
        .free = __com_arena_as_lg_allocator_free,
    };
}

#endif // ARENA_H_
