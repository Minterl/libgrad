#include "libgrad/internal/alloc.h"
#include <libgrad/internal/map.h>
#include <libgrad/internal/debug.h>

#include <stdint.h>

#define LG_MAP_EMPTY_SENTINEL UINT8_C(0x0)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// General utilities necessary for the table
///
////////////////////////////////////////////////////////////////////////////////

#define LG_U64_HAS_ZERO_BYTE(x) ((((x) - UINT64_C(0x0101010101010101)) & ~(x) & UINT64_C(0x8080808080808080)) != 0)

#define LG__MMH_C1 0xcc9e2d51u
#define LG__MMH_C2 0x1b873593u
#define LG__MMH_C3 0x85ebca6bu
#define LG__MMH_C4 0xc2b2ae35u
#define LG__MMH_R1 15u
#define LG__MMH_R2 13u
#define LG__MMH_M  5u
#define LG__MMH_N  0xe6546b64u
#define LG__MMH_S  0u

#define LG__MMH_ROL(x, width, bits) (((x) << (bits)) | ((x) >> ((width) - (bits))))

LG_ALWAYS_INLINE
uint32_t LG__MurmurHash(uint32_t kh) {
    kh = LG__MMH_ROL(kh * LG__MMH_C1, 32, LG__MMH_R1);
    kh *= LG__MMH_C2;
    kh = LG__MMH_S ^ kh;
    kh = LG__MMH_ROL(kh, 32, LG__MMH_R2) * LG__MMH_M + LG__MMH_N;
    kh = kh ^ 4;
    kh = kh ^ (kh >> 16);
    kh = kh * LG__MMH_C3;
    kh = kh ^ (kh >> 13);
    kh = kh * LG__MMH_C4;
    kh = kh ^ (kh >> 16);
    return kh;
}

LG_ALWAYS_INLINE
size_t LG__NextPow2(size_t x) {
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

enum lg_status LG_MapInit(struct lg_map *map, struct lg_allocator *alloc, size_t cap) {
    cap = cap < 8 ? 8 : LG__NextPow2(cap);
    const size_t align = 16;

    const size_t sz_keys = cap * sizeof(uint64_t);
    const size_t sz_fingerprints = (cap / 8) * sizeof(uint64_t);

    uint8_t *ptrs[2] = {0};
    enum lg_status status = LG__AllocContiguousBlocks(
        alloc, 
        ptrs,
        NULL,
        (size_t[]){sz_keys, sz_fingerprints},
        2,
        align
    );
    if (status != LG_STATUS_OK) {
        return status;
    }

    map->cap = cap;
    map->keys = (uint64_t*)ptrs[0];
    map->fingerprints_as = (void*)(ptrs[1]);

    return LG_STATUS_OK;
}

void LG_MapDeinit(struct lg_map *map, struct lg_allocator *alloc) {
    alloc->Free(alloc->ctx, map->keys);
    LG__ZERO(map, sizeof(struct lg_map));
}

LG_ALWAYS_INLINE
void LG__MapMakeHash(uint64_t key, uint64_t *out_hash, uint8_t *out_fingerprint) {
    const uint64_t full_hash = LG__MurmurHash(key); // TODO: make hash actually 64 bits
    const size_t hash = full_hash & ~UINT8_C(0xF);
    const uint8_t fingerprint = (full_hash & UINT8_C(0xF)) | UINT8_C(0x1);

    LG__Assert(fingerprint != LG_MAP_EMPTY_SENTINEL);

    if (out_fingerprint != NULL) {
        *out_fingerprint = fingerprint;
    }
    if (out_hash != NULL) {
        *out_hash = hash;
    }
}

LG_ALWAYS_INLINE
enum lg_status LG__MapProbe(
    struct lg_map *map,
    uint64_t key,
    uint64_t hash,
    uint8_t fingerprint,
    size_t *LG_NULLABLE out_last_idx, 
    bool *LG_NULLABLE out_found
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

        const bool has_match = LG_U64_HAS_ZERO_BYTE(block ^ fingerprint_broadcasted);
        if (has_match) {
            for (size_t j = 0; j < 8; j++) {
                if (map->fingerprints_as[i].individual[j] == LG_MAP_EMPTY_SENTINEL) {
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
        const bool has_empty_sentinel = LG_U64_HAS_ZERO_BYTE(block);
        if (has_empty_sentinel) {
            for (size_t j = 0; j < 8; j++) {
                if (map->fingerprints_as[i].individual[j] == LG_MAP_EMPTY_SENTINEL) {
                    ret_last_idx = i * 8 + j;
                    goto out_success;
                }
            }
            LG__Unreachable();
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
    return LG_STATUS_OUT_OF_MEMORY;

out_success:
    if (out_last_idx != NULL) {
        *out_last_idx = ret_last_idx;
    }
    if (out_found != NULL) {
        *out_found = ret_found;
    }
    return LG_STATUS_OK;
}

enum lg_status LG_MapEnsure(
    struct lg_map *map,
    uint64_t key,
    size_t *LG_NULLABLE out_idx,
    bool *LG_NULLABLE out_was_occupied
) {
    LG__Assert(LG__NextPow2(map->cap) == map->cap && map->cap >= 8);

    enum lg_status status = LG_STATUS_OK;

    uint64_t hash;
    uint8_t fingerprint;
    LG__MapMakeHash(key, &hash, &fingerprint);

    bool found;
    size_t last_idx;
    status = LG__MapProbe(map, key, hash, fingerprint, &last_idx, &found);
    if (status != LG_STATUS_OK) {
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

size_t LG_MapGet(struct lg_map *map, uint64_t key, bool *LG_NULLABLE out_found) {
    LG__Assert(LG__NextPow2(map->cap) == map->cap && map->cap >= 8);

    uint64_t hash;
    uint8_t fingerprint;
    LG__MapMakeHash(key, &hash, &fingerprint);

    bool found;
    size_t last_idx;
    enum lg_status status = LG__MapProbe(map, key, hash, fingerprint, &last_idx, &found);
    LG__Assert(status == LG_STATUS_OK); // we are't allocating a slot, and this only returns
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

void LG_MapIterInit(struct lg_map_iter *iter, struct lg_map *map) {
    LG__ZERO(iter, sizeof(struct lg_map_iter));
    iter->map = map;
}

LG_ALWAYS_INLINE
bool LG_MapIterAdvance(struct lg_map_iter *iter) {
    const size_t fingerprint_blocks_cap = iter->map->cap / 8;

    for (; iter->matrix_coord[0] < fingerprint_blocks_cap; iter->matrix_coord[0]++) {
        if (iter->map->fingerprints_as[iter->matrix_coord[0]].block == 0) {
            iter->matrix_coord[0]++;
            continue;
        }

        for (; iter->matrix_coord[1] < 8; iter->matrix_coord[1]++) {
            const uint8_t fingerprint = iter->map->fingerprints_as[iter->matrix_coord[0]].individual[iter->matrix_coord[1]];
            if (fingerprint != LG_MAP_EMPTY_SENTINEL) {
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
