#ifndef LG_VM_SYMTAB_H_
#define LG_VM_SYMTAB_H_

#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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
    size_t    cap_table;
    size_t    cap_array;
    size_t    next_array_idx;

    bool     *occupied    LG_CHECK_BOUNDS(cap_table);
    uint32_t *symbol_ids  LG_CHECK_BOUNDS(cap_table);
    size_t   *array_idxs  LG_CHECK_BOUNDS(cap_table);
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
    const size_t sz_array_idxs = LG__ALIGN_UP(cap * sizeof(size_t), align);

    uint8_t *ptrs[3] = {0};
    size_t bytes_allocated = 0;
    enum lg_status status = LG__AllocContiguousBlocks(
        alloc, 
        ptrs,
        &bytes_allocated,
        (size_t[]){sz_occupied, sz_symbol_ids, sz_array_idxs},
        3,
        align
    );
    if (status != LG_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < bytes_allocated; i++) {
        ptrs[0][i] = 0;
    }

    table->cap_table = cap;
    table->occupied = (bool*)ptrs[0];
    table->symbol_ids = (uint32_t*)ptrs[1];
    table->array_idxs = (size_t*)ptrs[2];

    return LG_STATUS_OK;
}

static inline void LG_VM__SymtabDeinit(struct lg_vm__symtab *table, struct lg_allocator *alloc) {
    alloc->Free(alloc->ctx, table->occupied);
    table->cap_table = 0;
    table->occupied = NULL;
    table->symbol_ids = NULL;
    table->array_idxs = NULL;
}

static inline enum lg_status LG_VM__SymtabUpsert(
    struct lg_vm__symtab *table,
    size_t *LG_NULLABLE out_idx,
    bool *LG_NULLABLE out_was_occupied,
    uint32_t symbol_id
) {
    if (table->cap_table == 0) {
        return LG_STATUS_NOT_FOUND;
    }

    const size_t start_idx = (table->cap_table <= 8) ? 0 : (LG_VM__MurmurHash(symbol_id) % table->cap_table);
    for (
        size_t i = start_idx, n_visited = 0;
        n_visited < table->cap_table;
        i = (i + 1) % table->cap_table, n_visited++
    ) {
        if (!table->occupied[i]) {
            table->occupied[i] = true;
            table->symbol_ids[i] = symbol_id;
            table->array_idxs[i] = table->next_array_idx;
            table->next_array_idx++;
            if (out_idx != NULL) {
                *out_idx = table->array_idxs[i];
            }
            if (out_was_occupied != NULL) {
                *out_was_occupied = false;
            }
            return LG_STATUS_OK;
        }
        if (table->symbol_ids[i] == symbol_id) {
            if (out_idx != NULL) {
                *out_idx = table->array_idxs[i];
            }
            if (out_was_occupied != NULL) {
                *out_was_occupied = true;
            }
            return LG_STATUS_OK;
        }
    }

    return LG_STATUS_OUT_OF_MEMORY;
}

#endif // LG_VM_SYMTAB_H_
