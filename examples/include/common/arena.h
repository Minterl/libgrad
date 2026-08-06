#ifndef ARENA_H_
#define ARENA_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include <libgrad/libgrad.h>

typedef struct
Arena {
    size_t cap;
    size_t offset;
    uint8_t *buf;
} Arena;

static inline uint8_t ArenaInit(Arena *a, size_t cap) {
    a->buf = malloc(cap);
    if (a->buf == NULL) {
        return -1;
    }
    a->cap = cap;
    a->offset = 0;
    return 0;
}

static inline void ArenaDestroy(Arena *a) {
    free(a->buf);
}

static inline uint8_t *ArenaAlloc(Arena *a, size_t size_bytes) {
    if (a->offset + size_bytes > a->cap) {
        return NULL;
    }
    uint8_t *ret = a->buf + a->offset;
    a->offset += size_bytes;
    return ret;
}

static inline void ArenaReset(Arena *a) {
    a->offset = 0; 
}

static inline void *__ArenaAsLgAllocatorAlloc(void *ctx, size_t size_bytes) {
    Arena *const a = ctx;
    return (void*)ArenaAlloc(a, size_bytes);
}

static inline void __ArenaAsLgAllocatorFree(void *ctx, void *ptr) {
    (void)ctx;
    (void)ptr;
}

static inline LG_Allocator ArenaAsLgAllocator(Arena *a) {
    return (LG_Allocator) {
        .ctx = a,
        .alloc = __ArenaAsLgAllocatorAlloc,
        .free = __ArenaAsLgAllocatorFree,
    };
}

#endif // ARENA_H_
