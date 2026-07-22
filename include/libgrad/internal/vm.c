#include <libgrad/internal/vm_symtab.h>
#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>
#include <libgrad/internal/debug.h>

struct lg_ir_symbol LG_IR__CreateSymbol(struct lg_ir_expr *expr) {
    const size_t id = expr->next_symbol_id;
    expr->next_symbol_id++;
    struct lg_ir_symbol s = {
        .id = id,
    };
    return s;
}

enum lg_status LG_IR__ExprAppendNode(
    struct lg_ir_expr *expr,
    const struct lg_ir_expr_node node 
) {
    if (expr->nodes_len + 1 >= expr->nodes_cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
    size_t next_idx = expr->nodes_len;
    expr->nodes_len += 1;
    expr->nodes[next_idx] = node;
    return LG_STATUS_OK;
}

enum lg_status LG_IR_BuftabInsert(struct lg_ir_expr *expr, uint32_t id, size_t bytes_required) {
    if (expr->buf_table_len >= expr->buf_table_cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
    for (size_t i = 0; i < expr->buf_table_len; i++) {
        if (expr->buf_table_ids[i] == id) {
            return LG_STATUS_DUPLICATE_KEY;
        }
    }
    size_t next_idx = expr->buf_table_len;
    expr->buf_table_len += 1;
    expr->buf_table_ids[next_idx] = id;
    expr->buf_table_bytes_required[next_idx] = bytes_required;
    return LG_STATUS_OK;
}

enum lg_status LG_IR_BuftabGetIdx(size_t *LG_NULLABLE out_idx, const struct lg_ir_expr *expr, uint32_t id) {
    for (size_t i = 0; i < expr->buf_table_len; i++) {
        if (expr->buf_table_ids[i] == id) {
            if (out_idx != NULL) {
                *out_idx = i;
            }
            return LG_STATUS_OK;
        }
    }
    return LG_STATUS_NOT_FOUND;
}

enum lg_status LG_IR_CreateSourceSymbol(
    struct lg_ir_symbol *out_symbol,
    struct lg_ir_expr *expr,
    uint32_t buf_id
) {
    if (out_symbol == NULL) {
        return LG_STATUS_INVALID_ARGUMENT;
    }

    size_t buf_idx = 0;
    enum lg_status status = LG_IR_BuftabGetIdx(&buf_idx, expr, buf_id);
    if (status != LG_STATUS_OK) {
        return status;
    }

    struct lg_ir_symbol sym = LG_IR__CreateSymbol(expr);
    status = LG_IR__ExprAppendNode(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_SOURCE,
        .x0_logical = sym,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    return LG_STATUS_OK;
}

enum lg_status LG_IR_AppendNop(struct lg_ir_expr *expr, struct lg_ir_symbol x) {
    return LG_IR__ExprAppendNode(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_NOP,
        .x0_logical = x,
    });
}

enum lg_status LG_IR_AppendAdd(
    struct lg_ir_expr *expr,
    struct lg_ir_symbol *y,
    const struct lg_ir_symbol x0,
    const struct lg_ir_symbol x1
) {
    struct lg_ir_symbol y_ = LG_IR__CreateSymbol(expr);
    enum lg_status status = LG_IR__ExprAppendNode(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_ADD,   
        .y_logical = y_,
        .x0_logical = x0,
        .x1_logical = x1,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }
    *y = y_;
    return LG_STATUS_OK;
}

enum lg_status LG_IR_AppendContract(
    struct lg_ir_expr *expr,
    struct lg_ir_symbol *y,
    struct lg_ir_symbol x0,
    struct lg_ir_symbol x1,
    size_t n_contracted_axes, 
    size_t n_batch_axes
) {
    struct lg_ir_symbol y_ = LG_IR__CreateSymbol(expr);
    enum lg_status status = LG_IR__ExprAppendNode(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_CONTRACT,   
        .y_logical = y_,
        .x0_logical = x0,
        .x1_logical = x1,
        .meta_as.contract.n_contracted_axes = n_contracted_axes,
        .meta_as.contract.n_batch_axes = n_batch_axes,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }
    *y = y_;
    return LG_STATUS_OK;
}

enum lg_status LG_IR__InferDims(struct lg_allocator *scratch, struct lg_ir_expr *expr) {
    enum lg_status status;

    struct lg_desc *symtab_descs = scratch->Alloc(scratch->ctx, expr->nodes_len);
    if (symtab_descs == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }
    struct lg_ir__symtab symtab = {0};
    status = LG_IR__SymtabInit(&symtab, scratch, expr->nodes_len * 2);
    if (status != LG_STATUS_OK) {
        goto out_free_descs;
    }

    for (size_t i = 0; i < expr->nodes_len; i++) {
        size_t rank;
        size_t dim[LG_MAX_RANK];

        switch (expr->nodes[i].opcode) {
        case LG_OPCODE_NOP:
        case LG_OPCODE_SOURCE:
            continue;

        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB: 
        {
            status = LG_InferBroadcastedDims(
                &rank,
                dim, 
                (const struct lg_desc*[2]){
                    &expr->nodes[i].x0_physical,
                    &expr->nodes[i].x1_physical,
                },
                2
            );
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_CONTRACT:
        {
            LG_InferContractedDims(
                &rank,
                dim, 
                &expr->nodes[i].x0_physical,
                &expr->nodes[i].x1_physical,
                expr->nodes[i].meta_as.contract.n_contracted_axes,
                expr->nodes[i].meta_as.contract.n_batch_axes
            );
            break;
        }

        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
            status = LG_STATUS_UNSUPPORTED_OPCODE;
            goto out_deinit_symtab;
        }

        size_t idx = 0;
        status = LG_IR__SymtabUpsert(&symtab, &idx, NULL, expr->nodes[i].y_logical.id);
        if (status != LG_STATUS_OK) {
            LG__Unreachable("should compute max size properly");
            goto out_deinit_symtab;
        }
        symtab_descs[idx].rank = rank;
        for (size_t j = 0; j < LG_MAX_RANK; j++) {
            symtab_descs[idx].dim[j] = dim[j];
        }
    }

    for (size_t i_node = 0; i_node < expr->nodes_len; i_node++) {
        const uint32_t symbol_ids[3] = {
            expr->nodes[i_node].y_logical.id,
            expr->nodes[i_node].x0_logical.id,
            expr->nodes[i_node].x1_logical.id,
        };
        struct lg_desc *const node_descs[3] = {
            &expr->nodes[i_node].y_physical,
            &expr->nodes[i_node].x0_physical,
            &expr->nodes[i_node].x1_physical,
        };
        for (size_t i_sym = 0; i_sym < 3; i_sym++) {
            size_t idx = 0;
            status = LG_IR__SymtabUpsert(&symtab, &idx, NULL, symbol_ids[i_sym]);
            if (status != LG_STATUS_OK) {
                LG__Unreachable("should compute max size properly");
                goto out_deinit_symtab;
            }
            node_descs[idx]->rank = symtab_descs[idx].rank;
            for (size_t j = 0; j < LG_MAX_RANK; j++) {
                node_descs[i_sym]->dim[j] = symtab_descs[idx].dim[j];
            }
        }
    }

out_deinit_symtab:
    LG_IR__SymtabDeinit(&symtab, scratch);
out_free_descs:
    scratch->Free(scratch->ctx, symtab_descs);

    return status;
}

enum lg_status LG_IR__AssignLayouts(struct lg_ir_expr *expr, enum lg_layout layout, size_t unit_align) {
    for (size_t i_node = 0; i_node < expr->nodes_len; i_node++) {
        if (expr->nodes[i_node].opcode == LG_OPCODE_NOP) {
            continue;
        }
        struct lg_desc *const descs[3] = {
            &expr->nodes[i_node].y_physical,
            &expr->nodes[i_node].x0_physical,
            &expr->nodes[i_node].x1_physical,
        };
        for (size_t i_desc = 0; i_desc < 3; i_desc++) {
            struct lg_desc *desc = descs[i_desc];
            for (size_t i_stride = 0; i_stride < LG_MAX_RANK; i_stride++) {
                if (desc->strides[i_stride] != 0) {
                    goto skip_layout;
                }
            }
            enum lg_status status = LG_DescComputeStrides(desc, layout, unit_align);
            if (status != LG_STATUS_OK) {
                return status;
            }
skip_layout:;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR__Bufferize(
    struct lg_allocator *scratch,
    struct lg_ir_expr *expr,
    size_t *LG_NULLABLE out_bytes_required,
    uint32_t buf_id,
    size_t align
) {
    // TODO: we need a better representation for unaries
    
    enum lg_status status = LG_STATUS_OK;

    // --- Initialize symbol table & zero pools ---
    uint8_t *scratch_bufs[4] = {0};
    size_t bytes_allocated = 0;
    status = LG__AllocContiguousBlocks(
        scratch,
        scratch_bufs,
        &bytes_allocated,
        (size_t[]){
            expr->nodes_len * sizeof(size_t),
            expr->nodes_len * sizeof(size_t),
            expr->nodes_len * sizeof(size_t),
            expr->nodes_len * sizeof(size_t),
        },
        4,
        8
    );
    if (status != LG_STATUS_OK) {
        return status;
    }
    size_t *const restrict symtab_sizes = (size_t*)scratch_bufs[0];
    size_t *const restrict symtab_dead_after = (size_t*)scratch_bufs[1];
    size_t *const restrict total_freed_after_time = (size_t*)scratch_bufs[2];
    size_t *const restrict symtab_offsets = (size_t*)scratch_bufs[3];

    struct lg_ir__symtab symtab = {0};
    status = LG_IR__SymtabInit(&symtab, scratch, expr->nodes_len * 2);
    if (status != LG_STATUS_OK) {
        goto out_free_scratch_bufs;
    }

    for (size_t i_time = 0; i_time < expr->nodes_len; i_time++) {
        const uint32_t symbol_ids[3] = {
            expr->nodes[i_time].y_logical.id,
            expr->nodes[i_time].x0_logical.id,
            expr->nodes[i_time].x1_logical.id,
        };
        for (size_t i_sym = 0; i_sym < 3; i_sym++) {
            size_t idx = 0;
            status = LG_IR__SymtabUpsert(&symtab, &idx, NULL, symbol_ids[i_sym]);
            if (status != LG_STATUS_OK) {
                LG__Unreachable("should compute max size properly");
                goto out_deinit_symtab;
            }
            symtab_sizes[idx] = 0;
            symtab_dead_after[idx] = i_time;
            symtab_offsets[idx] = 0;
        }
        total_freed_after_time[i_time] = 0;
    }

    // --- Calculate physical sizes & map them to timesteps ---
    for (size_t i_time = 0; i_time < expr->nodes_len; i_time++) {
        const uint32_t symbol_ids[3] = {
            expr->nodes[i_time].y_logical.id,
            expr->nodes[i_time].x0_logical.id,
            expr->nodes[i_time].x1_logical.id,
        };
        const struct lg_desc *const descs[3] = {
            &expr->nodes[i_time].y_physical,
            &expr->nodes[i_time].x0_physical,
            &expr->nodes[i_time].x1_physical,
        };
        const uint32_t buf_ids[3] = {
            expr->nodes[i_time].y_buf_id,
            expr->nodes[i_time].x0_buf_id,
            expr->nodes[i_time].x1_buf_id,
        };
        for (size_t i_sym = 0; i_sym < 3; i_sym++) {
            size_t idx = 0;
            status = LG_IR__SymtabUpsert(&symtab, &idx, NULL, symbol_ids[i_sym]);
            if (status != LG_STATUS_OK) {
                LG__Unreachable("should compute max size properly");
                goto out_deinit_symtab;
            }
            if (buf_ids[i_sym] == buf_id) {
                const size_t size = LG_DescSizeInBytes(*descs[i_sym]);
                symtab_sizes[idx] = LG__ALIGN_UP(size, align);
            }
        }
    }
    for (size_t i = 0; i < symtab.cap_table; i++) {
        if (!symtab.occupied[i]) {
            continue;
        }
        const size_t idx = symtab.array_idxs[i];
        total_freed_after_time[symtab_dead_after[idx]] += symtab_sizes[idx];
    }

    // --- Core LSRA ---
    size_t current_offset = 0;
    size_t max_offset = 0;
    for (size_t i_time = 0; i_time < expr->nodes_len; i_time++) {
        size_t y_idx = 0;
        status = LG_IR__SymtabUpsert(&symtab, &y_idx, NULL, expr->nodes[i_time].y_logical.id);
        if (status != LG_STATUS_OK) {
            LG__Unreachable("should compute max size properly");
            goto out_deinit_symtab;
        }
        symtab_offsets[y_idx] = current_offset;
        current_offset += symtab_sizes[y_idx];
        if (current_offset > max_offset) {
            max_offset = current_offset;
        }
        current_offset -= total_freed_after_time[i_time];
    }

    if (out_bytes_required != NULL) {
        *out_bytes_required = max_offset;
    }

    // --- Finally, assign offsets to IR nodes ---
    for (size_t i_time = 0; i_time < expr->nodes_len; i_time++) {
        const uint32_t symbol_ids[3] = {
            expr->nodes[i_time].y_logical.id,
            expr->nodes[i_time].x0_logical.id,
            expr->nodes[i_time].x1_logical.id,
        };
        size_t *const node_offsets[3] = {
            &expr->nodes[i_time].y_offset,
            &expr->nodes[i_time].x0_offset,
            &expr->nodes[i_time].x1_offset,
        };
        const uint32_t buf_ids[3] = {
            expr->nodes[i_time].y_buf_id,
            expr->nodes[i_time].x0_buf_id,
            expr->nodes[i_time].x1_buf_id,
        };
        for (size_t i_sym = 0; i_sym < 3; i_sym++) {
            size_t idx = 0;
            status = LG_IR__SymtabUpsert(&symtab, &idx, NULL, symbol_ids[i_sym]);
            if (status != LG_STATUS_OK) {
                LG__Unreachable("isn't inserting");
                goto out_deinit_symtab;
            }
            if (buf_ids[i_sym] == buf_id) {
                *node_offsets[i_sym] = symtab_offsets[idx];
            }
        }
    }

out_deinit_symtab:
    LG_IR__SymtabDeinit(&symtab, scratch);
out_free_scratch_bufs:
    scratch->Free(scratch->ctx, scratch_bufs[0]);
    return status;
}

enum lg_status LG_IR__SortAxes(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->nodes_len; i++) {
        status = LG_SortAxes((struct lg_desc*[]){
            &expr->nodes[i].y_physical,
            &expr->nodes[i].x0_physical,
            &expr->nodes[i].x1_physical,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR__CoalesceAxes(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->nodes_len; i++) {
        status = LG_CoalesceAxes((struct lg_desc*[]){
            &expr->nodes[i].y_physical,
            &expr->nodes[i].x0_physical,
            &expr->nodes[i].x1_physical,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR_CompileExpr(
    size_t *LG_NULLABLE out_bytes_required,
    struct lg_allocator *scratch,
    struct lg_ir_expr *expr,
    size_t mem_align
) {
    enum lg_status status;
    status = LG_IR__InferDims(scratch, expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__AssignLayouts(expr, LG_LAYOUT_ROW_MAJOR /* TODO */, mem_align);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__Bufferize(scratch, expr, out_bytes_required, 0 /* TODO */, mem_align);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__SortAxes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__CoalesceAxes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    return LG_STATUS_OK;
}

enum lg_status LG_AllocExpr(
    struct lg_allocator *perm,
    uint8_t *LG_NULLABLE *out_ptr,
    size_t *LG_NULLABLE out_bytes_allocated,
    struct lg_ir_expr *expr,
    size_t nodes_cap,
    size_t bufmap_cap
) {
    uint8_t *ptrs[3] = {0};
    size_t bytes_allocated = 0;
    enum lg_status status = LG__AllocContiguousBlocks(
        perm,
        ptrs, 
        &bytes_allocated,
        (size_t[]){
            nodes_cap * sizeof(struct lg_ir_expr_node),
            bufmap_cap * sizeof(struct lg_ir_expr_node),
            bufmap_cap * sizeof(struct lg_ir_expr_node),
        },
        3,
        16
    );
    if (status != LG_STATUS_OK) {
        return status; 
    }

    for (size_t i = 0; i < bytes_allocated; i++) {
        ptrs[0][i] = 0;
    }

    expr->nodes = (struct lg_ir_expr_node*)(ptrs[0]);
    expr->nodes_cap = nodes_cap;
    expr->nodes_len = 0;
    expr->buf_table_ids = (uint32_t*)(ptrs[1]);
    expr->buf_table_bytes_required = (size_t*)(ptrs[2]);
    expr->buf_table_cap = bufmap_cap;
    expr->buf_table_len = 0;

    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = bytes_allocated;
    }
    if (out_ptr != NULL) {
        *out_ptr = ptrs[0];
    }

    return LG_STATUS_OK;
}

void LG_FreeExpr(struct lg_allocator *allocator, struct lg_ir_expr *expr) {
    allocator->Free(allocator->ctx, expr->nodes);
    expr->nodes_cap = 0;
    expr->nodes_len = 0;
    expr->nodes = NULL;
}
