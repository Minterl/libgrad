#ifndef ARENA_H_
#define ARENA_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include <libgrad/libgrad.h>

struct arena {
    size_t cap;
    size_t offset;
    uint8_t *buf;
};

static inline uint8_t arenaInit(struct arena *a, size_t cap) {
    a->buf = malloc(cap);
    if (a->buf == NULL) {
        return -1;
    }
    a->cap = cap;
    a->offset = 0;
    return 0;
}

static inline void arenaDestroy(struct arena *a) {
    free(a->buf);
}

static inline uint8_t *arenaAlloc(struct arena *a, size_t size_bytes) {
    if (a->offset + size_bytes > a->cap) {
        return NULL;
    }
    uint8_t *ret = a->buf + a->offset;
    a->offset += size_bytes;
    return ret;
}

static inline void arenaReset(struct arena *a) {
    a->offset = 0; 
}

static inline void *__arenaAsLgAllocatorAlloc(void *ctx, size_t size_bytes) {
    struct arena *const a = ctx;
    return (void*)arenaAlloc(a, size_bytes);
}

static inline void __arenaAsLgAllocatorFree(void *ctx, void *ptr) {
    (void)ctx;
    (void)ptr;
}

static inline struct lg_allocator arenaAsLgAllocator(struct arena *a) {
    return (struct lg_allocator) {
        .ctx = a,
        .alloc = __arenaAsLgAllocatorAlloc,
        .free = __arenaAsLgAllocatorFree,
    };
}

#endif // ARENA_H_
