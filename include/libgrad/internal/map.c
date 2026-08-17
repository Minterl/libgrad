#include <libgrad/internal/map.h>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// General utilities necessary for the table
///
////////////////////////////////////////////////////////////////////////////////

typedef enum 
LG_MapSentinel {
    LG_MapSentinel_Empty = UINT8_C(0x0),
} LG_MapSentinel;

#define LG_MMH_C1 0xcc9e2d51u
#define LG_MMH_C2 0x1b873593u
#define LG_MMH_C3 0x85ebca6bu
#define LG_MMH_C4 0xc2b2ae35u
#define LG_MMH_R1 15u
#define LG_MMH_R2 13u
#define LG_MMH_M  5u
#define LG_MMH_N  0xe6546b64u
#define LG_MMH_S  0u

#define lg_mmh_rol(x, width, bits) (((x) << (bits)) | ((x) >> ((width) - (bits))))
#define lg_u64_has_zero_byte(x) ((((x) - UINT64_C(0x0101010101010101)) & ~(x) & UINT64_C(0x8080808080808080)) != 0)

lg_force_inline uint32_t 
lg_mmh(uint64_t key) {
    uint32_t hash = LG_MMH_S;
    for (uint8_t i = 0; i < 2; i++) {
        uint32_t chunk = key >> (32 * i);
        chunk = lg_mmh_rol(chunk * LG_MMH_C1, 32, LG_MMH_R1);
        chunk *= LG_MMH_C2;
        hash = LG_MMH_S ^ hash;
        hash = lg_mmh_rol(hash, 32, LG_MMH_R2) * LG_MMH_M + LG_MMH_N;
        hash = hash ^ 4;
        hash = hash ^ (hash >> 16);
        hash = hash * LG_MMH_C3;
        hash = hash ^ (hash >> 13);
        hash = hash * LG_MMH_C4;
        hash = hash ^ (hash >> 16);
    }
    return hash;
}

lg_force_inline size_t 
lg_next_pow2(size_t x) {
    if (x == 0) {
        return 1;
    }
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// Map functions/interface definition 
///
////////////////////////////////////////////////////////////////////////////////

LG_StatusKind 
lg_map_init(LG_Map *map, LG_Arena *arena, size_t cap) {
    cap = cap < 8 ? 8 : lg_next_pow2(cap);
    const size_t align = 16;

    const size_t sz_keys = cap * sizeof(uint64_t);
    const size_t sz_fingerprints = (cap / 8) * sizeof(uint64_t);

    uint64_t *keys = (uint64_t*)lg_arena_alloc(arena, sz_keys, align);
    LG_MapFingerprintBlock *fingerprints = (LG_MapFingerprintBlock*)lg_arena_alloc(arena, sz_fingerprints, align);
    if (keys == NULL || fingerprints == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    map->cap = cap;
    map->keys = keys;
    map->fingerprints_as = fingerprints;

    return LG_StatusKind_OK;
}

lg_force_inline void 
lg_map_make_hash(uint64_t key, uint64_t *out_hash, uint8_t *out_fingerprint) {
    const uint64_t full_hash = lg_mmh(key);
    const size_t hash = full_hash & ~UINT8_C(0xF);
    const uint8_t fingerprint = (full_hash & UINT8_C(0xF)) | UINT8_C(0x1);

    lg_assert(fingerprint != LG_MapSentinel_Empty);

    if (out_fingerprint != NULL) {
        *out_fingerprint = fingerprint;
    }
    if (out_hash != NULL) {
        *out_hash = hash;
    }
}

lg_force_inline LG_StatusKind 
lg_map_probe(
    LG_Map *map,
    uint64_t key,
    uint64_t hash,
    uint8_t fingerprint,
    size_t *lg_nullable out_last_idx, 
    bool *lg_nullable out_found
) {
    // Be warned when working on this function,
    // the behavior of the other map functions are very tightly coupled to this one.

    const size_t fingerprint_blocks_cap = map->cap / 8;
    const size_t starting_fingerprint_block_idx = (hash % map->cap) / 8;

    const uint64_t fingerprint_broadcasted =
        (uint64_t)fingerprint       |
        (uint64_t)fingerprint << 8  |
        (uint64_t)fingerprint << 16 |
        (uint64_t)fingerprint << 24 |
        (uint64_t)fingerprint << 32 |
        (uint64_t)fingerprint << 40 |
        (uint64_t)fingerprint << 48 |
        (uint64_t)fingerprint << 56;

    size_t ret_last_idx = 0;
    bool ret_found = false;
    for (
        size_t i = starting_fingerprint_block_idx, n_visited = 0;
        n_visited < fingerprint_blocks_cap;
        i = (i + 1) & (map->cap - 1), n_visited++
    ) {
        const uint64_t block = map->fingerprints_as[i].block;

        const bool has_match = lg_u64_has_zero_byte(block ^ fingerprint_broadcasted);
        if (has_match) {
            for (size_t j = 0; j < 8; j++) {
                if (map->fingerprints_as[i].individual[j] == LG_MapSentinel_Empty) {
                    ret_last_idx = i * 8 + j;
                    goto out_success;
                }
                if (map->fingerprints_as[i].individual[j] != fingerprint) {
                    continue;
                }
                if (map->keys[i * 8 + j] == key) {
                    ret_last_idx = i * 8 + j;
                    ret_found = true;
                    goto out_success;
                }
            }
        }

        // TODO: @perf refactor to allow a get to not search for an
        // empty space after the zero sentintel is found
        // and also to look less stupid
        const bool has_empty_sentinel = lg_u64_has_zero_byte(block);
        if (has_empty_sentinel) {
            for (size_t j = 0; j < 8; j++) {
                if (map->fingerprints_as[i].individual[j] == LG_MapSentinel_Empty) {
                    ret_last_idx = i * 8 + j;
                    goto out_success;
                }
            }
            lg_unreachable();
        }
    }

    // If we got here, we have not
    // a) found a match, nor
    // b) found a single sentinel.
    // Thus, we must be out of capacity.
    if (out_last_idx != NULL) {
        *out_last_idx = 0;
    }
    if (out_found != NULL) {
        *out_found = false;
    }
    return LG_StatusKind_OutOfMemory;

out_success:
    if (out_last_idx != NULL) {
        *out_last_idx = ret_last_idx;
    }
    if (out_found != NULL) {
        *out_found = ret_found;
    }
    return LG_StatusKind_OK;
}

LG_StatusKind 
lg_map_ensure(
    LG_Map *map,
    uint64_t key,
    size_t *lg_nullable out_idx,
    bool *lg_nullable out_was_occupied
) {
    lg_assert(lg_next_pow2(map->cap) == map->cap && map->cap >= 8);

    LG_StatusKind status = LG_StatusKind_OK;

    uint64_t hash;
    uint8_t fingerprint;
    lg_map_make_hash(key, &hash, &fingerprint);

    bool found;
    size_t last_idx;
    status = lg_map_probe(map, key, hash, fingerprint, &last_idx, &found);
    if (status != LG_StatusKind_OK) {
        found = false;
        last_idx = 0;
        goto out;
    }

    if (!found) {
        const size_t outer_idx = last_idx / 8;
        const size_t inner_idx = last_idx % 8;
        map->fingerprints_as[outer_idx].individual[inner_idx] = fingerprint;
        map->keys[last_idx] = key;
    }

out:
    if (out_idx != NULL) {
        *out_idx = last_idx;
    }
    if (out_was_occupied != NULL) {
        *out_was_occupied = found;
    }
    return status;
}

size_t 
lg_map_get(LG_Map *map, uint64_t key, bool *lg_nullable out_found) {
    lg_assert(lg_next_pow2(map->cap) == map->cap && map->cap >= 8);

    uint64_t hash;
    uint8_t fingerprint;
    lg_map_make_hash(key, &hash, &fingerprint);

    bool found;
    size_t last_idx;
    LG_StatusKind status = lg_map_probe(map, key, hash, fingerprint, &last_idx, &found);
    lg_assert(status == LG_StatusKind_OK); // we are't allocating a slot, and this only returns
                                           // not ok when we're out of capacity

    if (out_found != NULL) {
        *out_found = found;
    }

    return found ? last_idx : 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// Map iterator stuff
///
////////////////////////////////////////////////////////////////////////////////

void 
lg_map_iter_init(LG_MapIter *iter, LG_Map *map) {
    lg_memzero(iter, sizeof(LG_MapIter));
    iter->map = map;
}

lg_force_inline bool 
lg_map_iter_advance(LG_MapIter *iter) {
    const size_t fingerprint_blocks_cap = iter->map->cap / 8;

    for (; iter->matrix_coord[0] < fingerprint_blocks_cap; iter->matrix_coord[0]++) {
        if (iter->map->fingerprints_as[iter->matrix_coord[0]].block == 0) {
            iter->matrix_coord[0]++;
            continue;
        }

        for (; iter->matrix_coord[1] < 8; iter->matrix_coord[1]++) {
            const uint8_t fingerprint = iter->map->fingerprints_as[iter->matrix_coord[0]].individual[iter->matrix_coord[1]];
            if (fingerprint != LG_MapSentinel_Empty) {
                iter->idx = iter->matrix_coord[0] * 8 + iter->matrix_coord[1];
                iter->key = iter->map->keys[iter->matrix_coord[0] * 8 + iter->matrix_coord[1]];
                iter->matrix_coord[1]++;
                return true;
            }
        }

        iter->matrix_coord[1] = 0;
    }

    return false;
}
