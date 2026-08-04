#ifndef LIBGRAD_IMPLEMENTATION
#define LIBGRAD_IMPLEMENTATION
#endif // LIBGRAD_IMPLEMENTATION
#include <libgrad/libgrad.h>
#include <libgrad/internal/map.h>

#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#include "testing.h"

void *alloc_libc(void *_, size_t bytes) {
    (void)_;
    return calloc(bytes, 1);
}

void free_libc(void* _, void *ptr) {
    (void)_;
    return free(ptr);
}

static struct lg_allocator libc_alloc = {
    .Alloc = alloc_libc,
    .Free = free_libc,
};

test_status test_map() {
    enum lg_status status;

    struct lg_map map = {0};
    status = LG_MapInit(&map, &libc_alloc, 32);
    test_assert(status == LG_STATUS_OK, "failed to init map");

    bool was_occupied;

    size_t first_idx;
    {
        status = LG_MapEnsure(&map, 1234, &first_idx, &was_occupied);
        test_assert(status == LG_STATUS_OK, "failed to insert into map");
        test_assert(!was_occupied, "incorrectly indicated the slot was occpuied");

        size_t idx_1;
        status = LG_MapEnsure(&map, 1234, &idx_1, &was_occupied);
        test_assert(status == LG_STATUS_OK, "failed to upsert into/get from map");
        test_assert(was_occupied, "incorrectly indicated the slot not was occpuied");
        test_assert(first_idx == idx_1, "gave wrong index");

        size_t idx_2 = LG_MapGet(&map, 1234, &was_occupied);
        test_assert(was_occupied, "incorrectly indicated the slot not was occpuied");
        test_assert(first_idx == idx_2, "gave wrong index");
    }

    {
        size_t idx_0;
        status = LG_MapEnsure(&map, 4321, &idx_0, &was_occupied);
        test_assert(status == LG_STATUS_OK, "failed to insert into map");
        test_assert(!was_occupied, "incorrectly indicated the slot was occpuied");
        test_assert(idx_0 != first_idx, "clobbered original slot; idx: %lu", idx_0);

        size_t idx_1;
        status = LG_MapEnsure(&map, 4321, &idx_1, &was_occupied);
        test_assert(status == LG_STATUS_OK, "failed to upsert into/get from map");
        test_assert(was_occupied, "incorrectly indicated the slot not was occpuied");
        test_assert(idx_0 == idx_1, "gave wrong index");

        size_t idx_2 = LG_MapGet(&map, 4321, &was_occupied);
        test_assert(was_occupied, "incorrectly indicated the slot not was occpuied");
        test_assert(idx_0 == idx_2, "gave wrong index");
    }

    LG_MapDeinit(&map, &libc_alloc);

    return TEST_STATUS_OK;
}

int main() {
    test_run(map);
    return 0;
}
