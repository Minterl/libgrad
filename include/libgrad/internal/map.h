#ifndef LG_MAP_H_
#define LG_MAP_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

#include <stdint.h>

struct lg_map {
    size_t cap;

    union {
        uint8_t individual[8];
        uint64_t block;
    } *fingerprints_as;
    uint64_t *keys;
};

enum lg_status LG_MapInit(struct lg_map *map, struct lg_allocator *alloc, size_t cap);
void LG_MapDeinit(struct lg_map *map, struct lg_allocator *alloc);

/// Ensures a key is present inside the map
/// Cannot realloc memory
enum lg_status LG_MapEnsure(struct lg_map *map, uint64_t key, size_t *LG_NULLABLE out_idx, bool *LG_NULLABLE out_was_occupied);

/// Returns the index corresponding to the key in the map, otherwise
/// zero.
///
/// Since zero may be a valid index, use `out_found` to determine whether
/// an entry was found.
size_t LG_MapGet(struct lg_map *map, uint64_t key, bool *LG_NULLABLE out_found);

#endif // LG_MAP_H_
