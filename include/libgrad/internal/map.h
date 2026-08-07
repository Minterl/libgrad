#ifndef LG_MAP_H_
#define LG_MAP_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

#include <stdint.h>

/// Implements a swiss map-like slot map.
typedef struct
LG_Map {
    /// Must be a power of two, and will be implicitly rounded to one 
    /// during initialization.
    size_t cap;

    union {
        uint8_t individual[8];
        uint64_t block;
    } *fingerprints_as;

    uint64_t *keys lg_check_bounds(cap);
} LG_Map;

typedef struct
LG_MapIter {
    LG_Map *map;

    uint32_t key;
    size_t idx;

    /// This treats the map fingerprints array as a matrix in R^2
    /// with dims {8 x `iter.map.cap` / 8} i.e there are 8 fingerprints per
    /// u64 in the map key list.
    size_t matrix_coord[2];
} LG_MapIter;

LG_StatusKind 
lg_map_init(LG_Map *map, LG_Allocator *alloc, size_t cap);

void 
lg_map_deinit(LG_Map *map, LG_Allocator *alloc);

/// Ensures a key is present inside the map
/// Cannot realloc memory
LG_StatusKind 
lg_map_ensure(LG_Map *map, uint64_t key, size_t *lg_nullable out_idx, bool *lg_nullable out_was_occupied);

/// Returns the index corresponding to the key in the map, otherwise
/// zero.
///
/// Since zero may be a valid index, use `out_found` to determine whether
/// an entry was found.
size_t 
lg_map_get(LG_Map *map, uint64_t key, bool *lg_nullable out_found);

void 
lg_map_iter_init(LG_MapIter *iter, LG_Map *map);

/// Returns false once the iterator is finished.
lg_force_inline bool 
lg_map_iter_advance(LG_MapIter *iter);

#endif // LG_MAP_H_
