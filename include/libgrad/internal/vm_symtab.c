#include <libgrad/internal/vm_symtab.h>

#define LG_IR__MMH_C1 0xcc9e2d51u
#define LG_IR__MMH_C2 0x1b873593u
#define LG_IR__MMH_C3 0x85ebca6bu
#define LG_IR__MMH_C4 0xc2b2ae35u
#define LG_IR__MMH_R1 15u
#define LG_IR__MMH_R2 13u
#define LG_IR__MMH_M  5u
#define LG_IR__MMH_N  0xe6546b64u
#define LG_IR__MMH_S  0u

#define LG_IR__MMH_ROL(x, width, bits) (((x) << (bits)) | ((x) >> ((width) - (bits))))

uint32_t LG_IR__MurmurHash(uint32_t kh) {
    kh = LG_IR__MMH_ROL(kh * LG_IR__MMH_C1, 32, LG_IR__MMH_R1);
    kh *= LG_IR__MMH_C2;
    kh = LG_IR__MMH_S ^ kh;
    kh = LG_IR__MMH_ROL(kh, 32, LG_IR__MMH_R2) * LG_IR__MMH_M + LG_IR__MMH_N;
    kh = kh ^ 4;
    kh = kh ^ (kh >> 16);
    kh = kh * LG_IR__MMH_C3;
    kh = kh ^ (kh >> 13);
    kh = kh * LG_IR__MMH_C4;
    kh = kh ^ (kh >> 16);
    return kh;
}

enum lg_status LG_IR__SymtabInit(struct lg_ir_symtab *table, struct lg_allocator *alloc, size_t cap) {
    const size_t align = 16;

    const size_t sz_occupied = cap * sizeof(bool);
    const size_t sz_symbol_ids = cap * sizeof(uint32_t);
    const size_t sz_array_idxs = cap * sizeof(size_t);
    const size_t sz_descs = cap * sizeof(struct lg_desc);
    const size_t sz_buffer_ids = cap * sizeof(uint32_t);
    const size_t sz_buffer_offsets = cap * sizeof(size_t);

    uint8_t *ptrs[6] = {0};
    size_t bytes_allocated = 0;
    enum lg_status status = LG__AllocContiguousBlocks(
        alloc, 
        ptrs,
        &bytes_allocated,
        (size_t[]){
            sz_occupied,
            sz_symbol_ids,
            sz_array_idxs,
            sz_descs,
            sz_buffer_ids,
            sz_buffer_offsets,
        },
        6,
        align
    );
    if (status != LG_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < bytes_allocated; i++) {
        ptrs[0][i] = 0;
    }

    table->table_cap = cap;
    table->array_cap = cap;
    table->occupied = (bool*)ptrs[0];
    table->symbol_ids = (uint32_t*)ptrs[1];
    table->array_idxs = (size_t*)ptrs[2];
    table->descs = (struct lg_desc*)ptrs[3];
    table->buffer_ids = (uint32_t*)ptrs[4];
    table->buffer_offsets = (size_t*)ptrs[5];

    return LG_STATUS_OK;
}

void LG_IR__SymtabDeinit(struct lg_ir_symtab *table, struct lg_allocator *alloc) {
    alloc->Free(alloc->ctx, table->occupied);
    alloc->Free(alloc->ctx, table->descs);
    LG__ZERO(table, sizeof(struct lg_ir_symtab));
}

enum lg_status LG_IR__SymtabUpsert(
    struct lg_ir_symtab *table,
    size_t *LG_NULLABLE out_idx,
    bool *LG_NULLABLE out_was_occupied,
    uint32_t symbol_id
) {
    if (table->table_cap == 0) {
        return LG_STATUS_NOT_FOUND;
    }

    const size_t start_idx = (table->table_cap <= 8) ? (symbol_id % table->table_cap) : (LG_IR__MurmurHash(symbol_id) % table->table_cap);
    for (
        size_t i = start_idx, n_visited = 0;
        n_visited < table->table_cap;
        i = (i + 1) % table->table_cap, n_visited++
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
