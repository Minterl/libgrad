#ifndef LG_VM_SYMTAB_H_
#define LG_VM_SYMTAB_H_

#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct lg_ir__symtab {
    size_t    cap_table;
    size_t    cap_array;
    size_t    next_array_idx;

    bool     *occupied    LG_CHECK_BOUNDS(cap_table);
    uint32_t *symbol_ids  LG_CHECK_BOUNDS(cap_table);
    size_t   *array_idxs  LG_CHECK_BOUNDS(cap_table);
};

enum lg_status LG_IR__SymtabInit(struct lg_ir__symtab *table, struct lg_allocator *alloc, size_t cap);

void LG_IR__SymtabDeinit(struct lg_ir__symtab *table, struct lg_allocator *alloc);

enum lg_status LG_IR__SymtabUpsert(
    struct lg_ir__symtab *table,
    size_t *LG_NULLABLE out_idx,
    bool *LG_NULLABLE out_was_occupied,
    uint32_t symbol_id
);

#endif // LG_VM_SYMTAB_H_
