#ifndef LG_MAP_H_
#define LG_MAP_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

#include <stdint.h>

/// Implements a swiss map-like slot map.
struct lg_map {
    /// Must be a power of two, and will be implicitly rounded to one 
    /// during initialization.
    size_t cap;

    union {
        uint8_t individual[8];
        uint64_t block;
    } *fingerprints_as;

    uint64_t *keys LG_CHECK_BOUNDS(cap);
};

struct lg_map_iter {
    struct lg_map *map;

    uint32_t key;
    size_t idx;

    /// This treats the map fingerprints array as a matrix in R^2
    /// with dims {8 x `iter.map.cap` / 8} i.e there are 8 fingerprints per
    /// u64 in the map self.
    size_t matrix_coord[2];
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

void LG_MapIterInit(struct lg_map_iter *iter, struct lg_map *map);

/// Returns false once the iterator is finished.
LG_ALWAYS_INLINE
bool LG_MapIterAdvance(struct lg_map_iter *iter);

#endif // LG_MAP_H_
