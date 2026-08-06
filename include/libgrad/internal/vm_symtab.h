#ifndef LG_VM_SYMTAB_H_
#define LG_VM_SYMTAB_H_

#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct
LG_SymbolTable {
    size_t    table_cap;
    size_t    array_cap;
    size_t    n_symbols;

    bool      *occupied    lg_check_bounds(table_cap);
    uint32_t  *symbol_ids  lg_check_bounds(table_cap);
    size_t    *array_idxs  lg_check_bounds(table_cap);

    LG_StridedDesc  *descs           lg_check_bounds(array_cap);
    size_t          *buffer_offsets  lg_check_bounds(array_cap);
    uint32_t        *buffer_ids      lg_check_bounds(array_cap);
} LG_SymbolTable;

typedef struct
LG_SymbolTableIter {
    LG_SymbolTable *symtab;
    uint32_t symbol_id;
    size_t array_idx;
    
    size_t last_idx;
} LG_SymbolTableIter;

LG_StatusKind 
lg_ir_symtab_init(LG_SymbolTable *table, LG_Allocator *alloc, size_t cap);

void 
lg_ir_symtab_deinit(LG_SymbolTable *table, LG_Allocator *alloc);

LG_StatusKind 
lg_ir_symtab_upsert(
    LG_SymbolTable *table,
    size_t *lg_nullable out_idx,
    bool *lg_nullable out_was_occupied,
    uint32_t symbol_id
);

void 
lg_ir_symtab_iter_init(LG_SymbolTableIter *iter, LG_SymbolTable *symtab);

/// Returns false once there are no more entries to iterate.
lg_force_inline bool 
lg_ir_symtab_iter_advance(LG_SymbolTableIter *iter);

#endif // LG_VM_SYMTAB_H_
