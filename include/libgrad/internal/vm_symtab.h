#ifndef LG_VM_SYMTAB_H_
#define LG_VM_SYMTAB_H_

#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define LG_VM__SYMTAB_N_TRACKED_ARRAYS 4

#define LG_VM__MMH_C1 0xcc9e2d51u
#define LG_VM__MMH_C2 0x1b873593u
#define LG_VM__MMH_C3 0x85ebca6bu
#define LG_VM__MMH_C4 0xc2b2ae35u
#define LG_VM__MMH_R1 15u
#define LG_VM__MMH_R2 13u
#define LG_VM__MMH_M  5u
#define LG_VM__MMH_N  0xe6546b64u
#define LG_VM__MMH_S  0u

#define LG_VM__MMH_ROL(x, width, bits) (((x) << (bits)) | ((x) >> ((width) - (bits))))

struct lg_vm__symtab {
    size_t    cap;

    bool     *occupied                                       LG_CHECK_BOUNDS(cap);
    uint32_t *symbol_ids                                     LG_CHECK_BOUNDS(cap);
    size_t   (*array_idxs)[LG_VM__SYMTAB_N_TRACKED_ARRAYS]   LG_CHECK_BOUNDS(cap);
};

static inline uint32_t LG_VM__MurmurHash(uint32_t kh) {
    kh = LG_VM__MMH_ROL(kh * LG_VM__MMH_C1, 32, LG_VM__MMH_R1);
    kh *= LG_VM__MMH_C2;
    kh = LG_VM__MMH_S ^ kh;
    kh = LG_VM__MMH_ROL(kh, 32, LG_VM__MMH_R2) * LG_VM__MMH_M + LG_VM__MMH_N;
    kh = kh ^ 4;
    kh = kh ^ (kh >> 16);
    kh = kh * LG_VM__MMH_C3;
    kh = kh ^ (kh >> 13);
    kh = kh * LG_VM__MMH_C4;
    kh = kh ^ (kh >> 16);
    return kh;
}

static inline enum lg_status LG_VM__SymtabInit(struct lg_vm__symtab *table, struct lg_allocator *alloc, size_t cap) {
    const size_t align = 16;
    const size_t sz_occupied = LG__ALIGN_UP(cap * sizeof(bool), align);
    const size_t sz_symbol_ids = LG__ALIGN_UP(cap * sizeof(uint32_t), align);
    const size_t sz_array_idxs = LG__ALIGN_UP(cap * LG_VM__SYMTAB_N_TRACKED_ARRAYS * sizeof(size_t), align);
    const size_t sz_total = sz_occupied + sz_symbol_ids +sz_array_idxs;

    uint8_t *ptr = alloc->Alloc(alloc->ctx, sz_total);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < sz_total; i++) {
        ptr[i] = 0;
    }

    const uint8_t *occupied = ptr;
    const uint8_t *symbol_ids = occupied + sz_occupied;
    const uint8_t *array_idxs = symbol_ids + sz_symbol_ids;

    table->cap = cap;
    table->occupied = (bool*)occupied;
    table->symbol_ids = (uint32_t*)symbol_ids;
    table->array_idxs = (size_t(*)[LG_VM__SYMTAB_N_TRACKED_ARRAYS])array_idxs;

    return LG_STATUS_OK;
}

static inline void LG_VM__SymtabDeinit(struct lg_vm__symtab *table, struct lg_allocator *alloc) {
    alloc->Free(alloc->ctx, table->occupied);
    table->cap = 0;
    table->occupied = NULL;
    table->symbol_ids = NULL;
    table->array_idxs = NULL;
}

static inline size_t* LG_VM__SymtabUpsert(struct lg_vm__symtab *table, bool *out_was_occupied, uint32_t symbol_id) {
    if (table->cap == 0) {
        return NULL;
    }

    const size_t start_idx = (table->cap <= 8) ? 0 : (LG_VM__MurmurHash(symbol_id) % table->cap);
    for (
        size_t i = start_idx, n_visited = 0;
        n_visited < table->cap;
        i = (i + 1) % table->cap, n_visited++
    ) {
        if (!table->occupied[i]) {
            table->occupied[i] = true;
            table->symbol_ids[i] = symbol_id;
            for (size_t j = 0; j < LG_VM__SYMTAB_N_TRACKED_ARRAYS; j++) {
                table->array_idxs[i][j] = 0;
            }
            if (out_was_occupied != NULL) {
                *out_was_occupied = false;
            }
            return table->array_idxs[i];
        }
        if (table->symbol_ids[i] == symbol_id) {
            if (out_was_occupied != NULL) {
                *out_was_occupied = true;
            }
            return table->array_idxs[i];
        }
    }

    return NULL;
}

#endif // LG_VM_SYMTAB_H_
