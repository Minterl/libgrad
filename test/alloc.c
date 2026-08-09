#include <libgrad/internal/alloc.h>

#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#include "testing.h"

#include <stdio.h>
#include <stdlib.h>

void *alloc_libc(void *_, size_t bytes) {
    (void)_;
    return calloc(bytes, 1);
}

void free_libc(void* _, void *ptr) {
    (void)_;
    return free(ptr);
}

static LG_Allocator libc_alloc = {
    .alloc = alloc_libc,
    .free = free_libc,
    .default_slab_size_bytes = 128,
};

test_status test_alloc() {
    LG_Arena arena;
    lg_arena_init(&arena, &libc_alloc);

    LG_Scope scope = lg_push_scope(&arena);
    lg_arena_alloc(&arena, 32);
    lg_arena_alloc(&arena, 12);
    lg_pop_scope(&arena, scope);
    lg_arena_alloc(&arena, 2);
    lg_arena_alloc(&arena, 256);
    lg_arena_free_all(&arena);
    lg_arena_free_recycled(&arena);

    test_assert(arena.current_offset == 0, "current offset must have been reset");
    test_assert(arena.recycled_slabs_head == NULL, "there must be no recycled slabs left");
    test_assert(arena.current_slab == NULL, "there must be no active slabs left");

    return TEST_STATUS_OK;
}

int main() {
    test_run(alloc);
}

#include <libgrad/internal/alloc.c>
#include <libgrad/internal/debug.c>
