#include <libgrad/internal/vm_symtab.h>
#include <libgrad/internal/alloc.h>
#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>
#include <libgrad/internal/debug.h>
#include <libgrad/internal/strings.h>

#include <stdint.h>

enum lg_status LG_IR__CreateSymbol(struct lg_ir_compilation_context *ctx, struct lg_ir_symbol *out) {
    const size_t id = ctx->next_symbol_id;
    ctx->next_symbol_id++;
    *out = (struct lg_ir_symbol){
        .id = id,
    };
    bool was_occupied;
    enum lg_status status = LG_IR_SymtabUpsert(&ctx->symtab, NULL, &was_occupied, id);
    if (status != LG_STATUS_OK) {
        return status;
    }
    LG__Assert(!was_occupied);
    return LG_STATUS_OK;
}

enum lg_status LG_IR__ExprAppendNode(
    struct lg_ir_expr *expr,
    const struct lg_ir_expr_node node 
) {
    if (expr->nodes_len >= expr->nodes_cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
    size_t next_idx = expr->nodes_len;
    expr->nodes_len += 1;
    expr->nodes[next_idx] = node;
    return LG_STATUS_OK;
}

size_t LG_IR__ReportError_Write(void *ctx_, struct lg_string str) {
    struct lg_ir_compilation_context *ctx = ctx_;
    size_t bytes_written = LG_Strcpy((struct lg_string){
        .len = LG_IR_MAX_ERR_LEN - ctx->err_msg_len,
        .p = ctx->err_msg_backing_buf + ctx->err_msg_len,
    }, str);
    ctx->err_msg_len += bytes_written;
    return bytes_written;
}

// Reports error on a best-effort basis, filling the buffer as much as possible.
void LG_IR__ReportError(struct lg_ir_compilation_context *ctx, struct lg_string fmt, ...) {
    ctx->err_msg_len = 0;
    struct lg_writer w = {
        .ctx = (void*)ctx,
        .Write = LG_IR__ReportError_Write,
    };
    va_list ap;
    va_start(ap, fmt);
    enum lg_status _ = LG_VPrintf(&w, fmt, ap);
    (void)_;
    va_end(ap);
}

enum lg_status LG_IR_BuftabInsert(struct lg_ir_expr *expr, uint32_t id) {
    if (expr->buf_table_len >= expr->buf_table_cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
    for (size_t i = 0; i < expr->buf_table_len; i++) {
        if (expr->buf_table_ids[i] == id) {
            return LG_STATUS_DUPLICATE;
        }
    }
    size_t next_idx = expr->buf_table_len;
    expr->buf_table_len += 1;
    expr->buf_table_ids[next_idx] = id;
    expr->buf_table_bytes_required[next_idx] = 0;
    return LG_STATUS_OK;
}

enum lg_status LG_IR_BuftabGetIdx(const struct lg_ir_expr *expr, size_t *LG_NULLABLE out_idx, uint32_t id) {
    for (size_t i = 0; i < expr->buf_table_len; i++) 
    {
        if (expr->buf_table_ids[i] == id) {
            if (out_idx != NULL) {
                *out_idx = i;
            }
            return LG_STATUS_OK;
        }
    }
    return LG_STATUS_NOT_FOUND;
}

enum lg_status LG_IR_DeclareSource(
    struct lg_ir_compilation_context *ctx,
    struct lg_ir_symbol *out_symbol,
    struct lg_desc physical_desc,
    uint32_t buf_id
) {
    if (out_symbol == NULL) {
        return LG_STATUS_INVALID_ARGUMENT;
    }

    size_t buf_idx = 0;
    enum lg_status status = LG_IR_BuftabGetIdx(ctx->expr, &buf_idx, buf_id);
    if (status != LG_STATUS_OK) {
        return status;
    }

    struct lg_ir_symbol sym;
    status = LG_IR__CreateSymbol(ctx, &sym);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = LG_IR__ExprAppendNode(ctx->expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_SOURCE,
        .x0_logical = sym,
        .x0_physical = physical_desc,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    *out_symbol = sym;

    return LG_STATUS_OK;
}

enum lg_status LG_IR_DeclareSink(struct lg_ir_compilation_context *ctx, struct lg_ir_symbol sym) {
    for (size_t i = 0; i < ctx->expr->nodes_len; i++) {
        if (
            ctx->expr->nodes[i].y_logical.id == sym.id ||
            ctx->expr->nodes[i].x0_logical.id == sym.id ||
            ctx->expr->nodes[i].x1_logical.id == sym.id 
        ) {
            if (ctx->expr->nodes[i].opcode == LG_OPCODE_SINK) {
                return LG_STATUS_DUPLICATE;
            }
            enum lg_status status = LG_IR__ExprAppendNode(ctx->expr, (struct lg_ir_expr_node){
                .opcode = LG_OPCODE_SINK,
                .x0_logical = sym,
            });
            return status;
        }
    }

    return LG_STATUS_NOT_FOUND;
}

enum lg_status LG_IR_GetSinkLocation(
    uint32_t *LG_NULLABLE out_buf_id,
    size_t *LG_NULLABLE out_offset,
    struct lg_desc *LG_NULLABLE out_desc,
    struct lg_ir_symbol sym,
    struct lg_ir_expr *expr
) {
    uint32_t buf_id = 0;
    size_t offset = 0;
    struct lg_desc desc = {0};

    for (size_t i = 0; i < expr->nodes_len; i++) {
        if (expr->nodes[i].opcode != LG_OPCODE_SINK) {
            continue;
        }
        if (expr->nodes[i].y_logical.id == sym.id) {
            buf_id = expr->nodes[i].y_buf_id;
            offset = expr->nodes[i].y_offset;
            desc = expr->nodes[i].y_physical;
            goto found;
        }
        if (expr->nodes[i].x0_logical.id == sym.id) {
            buf_id = expr->nodes[i].x0_buf_id;
            offset = expr->nodes[i].x0_offset;
            desc = expr->nodes[i].x0_physical;
            goto found;
        }
        if (expr->nodes[i].x1_logical.id == sym.id) {
            buf_id = expr->nodes[i].x1_buf_id;
            offset = expr->nodes[i].x1_offset;
            desc = expr->nodes[i].x1_physical;
            goto found;
        }
    }
    return LG_STATUS_NOT_FOUND;

found:;
    if (out_buf_id != NULL) {
        *out_buf_id = buf_id;
    }
    if (out_offset != NULL) {
        *out_offset = offset;
    }
    if (out_desc != NULL) {
        *out_desc = desc;
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
    struct lg_ir_compilation_context *ctx,
    struct lg_ir_symbol *y,
    const struct lg_ir_symbol x0,
    const struct lg_ir_symbol x1
) {
    enum lg_status status;
    struct lg_ir_symbol y_;
    status = LG_IR__CreateSymbol(ctx, &y_);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__ExprAppendNode(ctx->expr, (struct lg_ir_expr_node){
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
    struct lg_ir_compilation_context *ctx,
    struct lg_ir_symbol *y,
    struct lg_ir_symbol x0,
    struct lg_ir_symbol x1,
    size_t n_contracted_axes, 
    size_t n_batch_axes
) {
    enum lg_status status;
    struct lg_ir_symbol y_;
    status = LG_IR__CreateSymbol(ctx, &y_);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__ExprAppendNode(ctx->expr, (struct lg_ir_expr_node){
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

enum lg_status LG_IR__ValidateExprStructure(struct lg_ir_compilation_context *ctx) {
    enum lg_status status = LG_STATUS_OK;
    struct lg_scratch_node *waypoint = LG__AcquireScratch(ctx->scratch);

    // Source/sink rules
    {
        bool sources_begin = false;
        bool sources_end = false;
        bool sinks_begin = false;
            
        for (size_t i = 0; i < ctx->expr->nodes_len; i++) {
            // Source declarations must be the first section of the ctx->expr,
            // while sink declarations must be at the end.
            if (ctx->expr->nodes[i].opcode == LG_OPCODE_SOURCE && !sources_begin) {
                if (i != 0) {
                    return LG_STATUS_INVALID_ARGUMENT;
                }
                sources_begin = true;
            } else if (ctx->expr->nodes[i].opcode == LG_OPCODE_SOURCE && sources_end) {
                return LG_STATUS_INVALID_ARGUMENT;
            } else if (ctx->expr->nodes[i].opcode != LG_OPCODE_SOURCE && sources_begin) {
                sources_end = true;
            }

            // Sink declarations must also always be followed by a sink
            // declaration or be the end of the ctx->expr
            if (sinks_begin && ctx->expr->nodes[i].opcode != LG_OPCODE_SINK) {
                return LG_STATUS_INVALID_ARGUMENT;
            }
        }

        if (sinks_begin && ctx->expr->nodes[ctx->expr->nodes_len - 1].opcode != LG_OPCODE_SINK) {
            return LG_STATUS_INVALID_ARGUMENT;
        }
    }

    // Scope validation
    {
        const size_t seen_ids_cap = ctx->expr->nodes_len * 3;
        size_t seen_ids_len = 0;
        uint32_t *seen_ids = (uint32_t*)LG__AllocScratch(ctx->scratch, &waypoint, seen_ids_cap * sizeof(uint32_t));
        if (seen_ids == NULL) {
            return LG_STATUS_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < ctx->expr->nodes_len; i++) {
            uint32_t new_id = 0;

            if (ctx->expr->nodes[i].opcode == LG_OPCODE_SOURCE) {
                new_id = ctx->expr->nodes[i].x0_logical.id;
                for (size_t j = 0; j < seen_ids_len; j++) {
                    if (seen_ids[j] == new_id) {
                        status = LG_STATUS_INVALID_ARGUMENT;
                        goto out_release_scratch;
                    }
                }
            } else {
                bool found_x0 = false;
                bool found_x1 = false;
                for (size_t j = 0; j < seen_ids_len; j++) {
                    if (seen_ids[j] == ctx->expr->nodes[i].x0_logical.id) {
                        found_x0 = true;
                    } else if (seen_ids[j] == ctx->expr->nodes[i].x1_logical.id) {
                        found_x1 = true;
                    }
                }

                if (!found_x0 || (!found_x1 && LG__OPCODE_IS_BINARY(ctx->expr->nodes[i].opcode))) {
                    status = LG_STATUS_INVALID_ARGUMENT;
                    goto out_release_scratch;
                }

                new_id = ctx->expr->nodes[i].y_logical.id;
            }

            LG__Assert(seen_ids_len + 1 <= seen_ids_cap);
            seen_ids[seen_ids_len] = new_id;
            seen_ids_len++;
        }
    }

out_release_scratch:
    LG__ReleaseScratch(ctx->scratch, &waypoint);
    return status;
}

enum lg_status LG_IR__InferDims(struct lg_ir_compilation_context *ctx) {
    enum lg_status status = LG_STATUS_OK;

    for (size_t i = 0; i < ctx->expr->nodes_len; i++) {
        if (ctx->expr->nodes[i].opcode == LG_OPCODE_SOURCE) {
            size_t symtab_idx;
            status = LG_IR_SymtabUpsert(&ctx->symtab, &symtab_idx, NULL, ctx->expr->nodes[i].x0_logical.id);
            LG__Assert(status == LG_STATUS_OK);
            ctx->symtab.descs[symtab_idx] = ctx->expr->nodes[i].x0_physical;
            continue;
        }

        bool was_occupied = false;
        size_t x0_symtab_idx = 0;
        size_t x1_symtab_idx = 0;
        status = LG_IR_SymtabUpsert(&ctx->symtab, &x0_symtab_idx, &was_occupied, ctx->expr->nodes[i].x0_logical.id);
        LG__Assert(status == LG_STATUS_OK);
        LG__Assert(was_occupied);
        if (LG__OPCODE_IS_BINARY(ctx->expr->nodes[i].opcode)) {
            status = LG_IR_SymtabUpsert(&ctx->symtab, &x1_symtab_idx, &was_occupied, ctx->expr->nodes[i].x1_logical.id);
            LG__Assert(status == LG_STATUS_OK);
            LG__Assert(was_occupied);
        }
        LG__Assert(x0_symtab_idx != x1_symtab_idx);

        size_t rank = 0;
        size_t dim[LG_MAX_RANK] = {0};

        switch (ctx->expr->nodes[i].opcode) {
        case LG_OPCODE_SOURCE:
            LG__Unreachable();
            continue;

        case LG_OPCODE_NOP:
        case LG_OPCODE_SINK:
            continue;

        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB: {
            status = LG_InferBroadcastedDims(
                &rank,
                dim, 
                (const struct lg_desc*[2]){
                    &ctx->symtab.descs[x0_symtab_idx],
                    &ctx->symtab.descs[x1_symtab_idx],
                },
                2
            );
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_CONTRACT: {
            status = LG_InferContractedDims(
                &rank,
                dim, 
                &ctx->symtab.descs[x0_symtab_idx],
                &ctx->symtab.descs[x1_symtab_idx],
                ctx->expr->nodes[i].meta_as.contract.n_contracted_axes,
                ctx->expr->nodes[i].meta_as.contract.n_batch_axes
            );
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
            LG__Unreachable("TODO");
        }

        size_t y_symtab_idx = 0;
        status = LG_IR_SymtabUpsert(&ctx->symtab, &y_symtab_idx, NULL, ctx->expr->nodes[i].y_logical.id);
        LG__Assert(status == LG_STATUS_OK);
        ctx->symtab.descs[y_symtab_idx].rank = rank;
        for (size_t j = 0; j < LG_MAX_RANK; j++) {
            ctx->symtab.descs[y_symtab_idx].dim[j] = dim[j];
        }
    }

    return LG_STATUS_OK;
}

void LG_IR__AssignLayouts(struct lg_ir_compilation_context *ctx, enum lg_layout layout, size_t unit_align) {
    for (size_t i_node = 0; i_node < ctx->expr->nodes_len; i_node++) {
        if (ctx->expr->nodes[i_node].opcode == LG_OPCODE_NOP) {
            continue;
        }
        struct lg_desc *const descs[3] = {
            &ctx->expr->nodes[i_node].y_physical,
            &ctx->expr->nodes[i_node].x0_physical,
            &ctx->expr->nodes[i_node].x1_physical,
        };
        for (size_t i_desc = 0; i_desc < 3; i_desc++) {
            struct lg_desc *desc = descs[i_desc];
            for (size_t i_stride = 0; i_stride < LG_MAX_RANK; i_stride++) {
                if (desc->strides[i_stride] != 0) {
                    goto skip_layout;
                }
            }
            enum lg_status status = LG_DescComputeStrides(desc, layout, unit_align);
            LG__Assert(status == LG_STATUS_OK);
skip_layout:;
        }
    }
}

LG_ALWAYS_INLINE 
void LG__WideSetBit(size_t len, uint64_t *inout_bitset, bool bit, size_t offset_rtl) {
    const size_t idx = len - 1 - (offset_rtl / 64);
    const size_t shift = offset_rtl % 64;
    LG__Assert(idx < len);
    if (bit) {
        inout_bitset[idx] |= UINT64_C(0x1) << shift;
    } else {
        inout_bitset[idx] &= ~(UINT64_C(0x1) << shift);
    }
}

LG_ALWAYS_INLINE 
bool LG__WideGetBit(size_t len, const uint64_t *bitset, size_t offset_rtl) {
    const size_t idx = len - 1 - (offset_rtl / 64);
    const size_t shift = offset_rtl % 64;
    LG__Assert(idx < len);
    return (bitset[idx] & (UINT64_C(0x1) << shift)) != 0;
}

LG_ALWAYS_INLINE 
void LG__WideOr(size_t len, uint64_t *restrict inout_bitset, uint64_t *b) {
    for (size_t i = 0; i < len; i++) {
        inout_bitset[i] |= b[i];
    }
}

LG_ALWAYS_INLINE 
void LG__WideAnd(size_t len, uint64_t *restrict inout_bitset, uint64_t *b) {
    for (size_t i = 0; i < len; i++) {
        inout_bitset[i] &= b[i];
    }
}

enum lg_status LG_IR__Bufferize(
    struct lg_ir_compilation_context *ctx,
    uint32_t buf_id,
    size_t align
) {
    struct size_table {
        uint32_t symbol_id;
        size_t symtab_array_idx;
        size_t size_bytes;
        size_t offset;
    };

    enum lg_status status = LG_STATUS_OK;
    struct lg_scratch_node *waypoint = LG__AcquireScratch(ctx->scratch);


    //////////////////////////////////////
    // ~~ Calculate physical sizes ~~

    size_t size_table_len = 0;
    struct size_table* const size_table = (struct size_table*)LG__AllocScratch(
        ctx->scratch,
        &waypoint,
        ctx->symtab.n_symbols * sizeof(struct size_table)
    );
    if (size_table == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    }
    // Construct the size table
    {
        struct lg_ir_symtab_iter iter = {0};
        LG_IR_SymtabIterInit(&iter, &ctx->symtab);
        while (LG_IR_SymtabIterAdvance(&iter)) {
            if (ctx->symtab.buffer_ids[iter.array_idx] != buf_id) {
                continue;
            }

            const size_t size = LG_DescSizeInBytes(ctx->symtab.descs[iter.array_idx]);
            const size_t size_aligned = LG__ALIGN_UP(size, align);

            size_table[size_table_len].symbol_id = iter.symbol_id;
            size_table[size_table_len].symtab_array_idx = iter.array_idx;
            size_table[size_table_len].size_bytes = size_aligned;
            size_table_len++;
        }
    }
    // Construct a sorted index map over the size table
    // TODO: @perf something other than bubble sort
    size_t *size_table_sorted_map = (size_t*)LG__AllocScratch(ctx->scratch, &waypoint, size_table_len * sizeof(size_t));
    if (size_table_sorted_map == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    } 
    for (size_t i = 0; i < size_table_len; i++) {
        size_table_sorted_map[i] = i;
    }
    for (size_t i = 0; i < size_table_len; i++) {
        bool swapped = false;
        for (size_t j = 1; j < size_table_len - i; j++) {
            const size_t idx_a = size_table_sorted_map[j - 1];
            const size_t idx_b = size_table_sorted_map[j];
            if (size_table[idx_a].size_bytes < size_table[idx_b].size_bytes) {
                const size_t temp = size_table_sorted_map[idx_b];
                size_table_sorted_map[idx_b] = size_table_sorted_map[idx_a];
                size_table_sorted_map[idx_a] = temp;
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }


    /////////////////////////////////////////
    // ~~ Construct the interval graph ~~
    
    const size_t row_elements = LG__ALIGN_UP(ctx->symtab.n_symbols, 64) / 64;
    const size_t mat_size = row_elements * ctx->symtab.n_symbols * sizeof(uint64_t);
    uint64_t *adj_matrix = (uint64_t*)LG__AllocScratch(ctx->scratch, &waypoint, mat_size);
    if (adj_matrix == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    } 
    {
        uint64_t *live_set = (uint64_t*)LG__AllocScratch(ctx->scratch, &waypoint, row_elements * sizeof(uint64_t));
        if (live_set == NULL) {
            status = LG_STATUS_OUT_OF_MEMORY;
            goto out_release_scratch;
        }
        for (size_t i_time = ctx->expr->nodes_len; i_time > 0; i_time--) {
            // When a symbol is used for the last time, it dies, meaning it will be live from now until it 
            // is declared (speaking in the reverse-temporal sense)
            size_t x0_idx = 0;
            status = LG_IR_SymtabUpsert(&ctx->symtab, &x0_idx, NULL, ctx->expr->nodes[i_time - 1].x0_logical.id);
            LG__Assert(status == LG_STATUS_OK);
            LG__WideSetBit(row_elements, live_set, true, x0_idx);

            size_t x1_idx = 0;
            status = LG_IR_SymtabUpsert(&ctx->symtab, &x1_idx, NULL, ctx->expr->nodes[i_time - 1].x1_logical.id);
            LG__Assert(status == LG_STATUS_OK);
            LG__WideSetBit(row_elements, live_set, true, x1_idx);

            // After the live set has been updated, we update the matrix
            struct lg_ir_symtab_iter iter = {0};
            LG_IR_SymtabIterInit(&iter, &ctx->symtab);
            while (LG_IR_SymtabIterAdvance(&iter)) {
                bool is_live = LG__WideGetBit(row_elements, live_set, iter.array_idx);
                if (is_live) {
                    LG__WideOr(row_elements, adj_matrix + (row_elements * iter.array_idx), live_set);
                }
            }

            // Just before a symbol is born, it is not alive
            // We do not kill the value until *after* updating the adjacency matrix b/c
            // the output needs a valid buffer during the operation.
            size_t y_idx = 0;
            status = LG_IR_SymtabUpsert(&ctx->symtab, &y_idx, NULL, ctx->expr->nodes[i_time - 1].y_logical.id);
            LG__Assert(status == LG_STATUS_OK);
            LG__WideSetBit(row_elements, live_set, false, y_idx);
        }
    }


    ////////////////////////////////////////////////////////
    // ~~ Best-fit greedy allocation ~~

    uint64_t *assigned_set = (uint64_t*)LG__AllocScratch(ctx->scratch, &waypoint, row_elements * sizeof(uint64_t));
    if (assigned_set == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    }
    uint64_t *assigned_and_live_set = (uint64_t*)LG__AllocScratch(ctx->scratch, &waypoint, row_elements * sizeof(uint64_t));
    if (assigned_and_live_set == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    }

    // `taken_ranges` is kind of a range set over the total memory space required by the block
    // at a given timestep.
    //
    // Each range [array[e], array[e + 1]) where e is on an even-or-zero integer index into the array
    // represents a range of offsets currently occupied by a value.
    // This is done so that any ranges [array[o], array[o + 1]), where o is an odd index into the array
    // represents a range of free offsets.
    //
    // Recording ranges in `taken_ranges` implicitly constructs `free_ranges`.
    const size_t free_ranges_cap = size_table_len * sizeof(size_t) * 2 + 1;
    size_t taken_ranges_len = 0;
    size_t *free_ranges = (size_t*)LG__AllocScratch(ctx->scratch, &waypoint, free_ranges_cap);
    if (free_ranges == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out_release_scratch;
    }
    size_t *const taken_ranges = free_ranges + 1;

    size_t total_size_bytes = 0;

    for (size_t i_unsorted = 0; i_unsorted < size_table_len; i_unsorted++) {
        const size_t i = size_table_sorted_map[i_unsorted];
        struct size_table *const this_symbol = &size_table[i];

        LG__MEMCPY(assigned_and_live_set, assigned_set, row_elements * sizeof(uint64_t));
        LG__WideAnd(row_elements, assigned_and_live_set, adj_matrix + (row_elements * i));

        taken_ranges_len = 0;
        LG__ZERO(free_ranges, free_ranges_cap);
        for (size_t j = 0; j < size_table_len; j++) {
            if (!LG__WideGetBit(row_elements, assigned_and_live_set, size_table[j].symtab_array_idx)) {
                continue;
            }

            taken_ranges[taken_ranges_len] = size_table[j].offset;
            taken_ranges_len++;
            taken_ranges[taken_ranges_len] = size_table[j].offset + size_table[j].size_bytes;
            taken_ranges_len++;
            LG__Assert(taken_ranges_len < free_ranges_cap - 1);
        }
        LG__Assert(taken_ranges_len % 2 == 0);

        // TODO: @perf something other than bubble sort
        for (size_t j = 0; j < taken_ranges_len; j += 2) {
            for (size_t k = 2; k < taken_ranges_len - j; k += 2) {
                if (taken_ranges[k - 2] > taken_ranges[k]) {
                    size_t temp_start = taken_ranges[k - 2];
                    size_t temp_end = taken_ranges[k - 1];
                    taken_ranges[k - 2] = taken_ranges[k];
                    taken_ranges[k - 1] = taken_ranges[k + 1];
                    taken_ranges[k] = temp_start;
                    taken_ranges[k + 1] = temp_end;
                }
            }
        }

        // Add the infnite space after the end of the array to complete the last
        // free rangec
        size_t free_ranges_len = taken_ranges_len + 1;
        free_ranges[free_ranges_len] = SIZE_MAX;
        free_ranges_len++;

        bool found_free_range = false;
        size_t minimum_free_range_found = SIZE_MAX;
        size_t offset = 0;
        for (
            size_t j = 0;
            j < free_ranges_len - 1;
            j += 2
        ) {
            const size_t lo = free_ranges[j];
            const size_t hi = free_ranges[j + 1];
            const size_t gap = hi - lo;
            LG__Assert(hi > lo);

            if (
                gap >= this_symbol->size_bytes &&
                gap < minimum_free_range_found
            ) {
                offset = free_ranges[j];
                found_free_range = true;
                minimum_free_range_found = gap;
                break;
            }
        }
        LG__Assert(found_free_range);

        this_symbol->offset = offset;

        if (this_symbol->offset + this_symbol->size_bytes > total_size_bytes) {
            total_size_bytes = this_symbol->offset + this_symbol->size_bytes;
        }

        LG__WideSetBit(row_elements, assigned_and_live_set, true, i_unsorted);
    }


    /////////////////////////////////////////////////////////////////////////
    // ~~ Write the offsets & buffer size back to the compilation state ~~

    for (size_t i = 0; i < size_table_len; i++) {
        size_t idx;
        bool occupied;
        status = LG_IR_SymtabUpsert(&ctx->symtab, &idx, &occupied, size_table[i].symbol_id);
        LG__Assert(status == LG_STATUS_OK);
        LG__Assert(occupied);

        ctx->symtab.buffer_offsets[idx] = size_table[i].offset;
    }

    size_t buftab_idx;
    status = LG_IR_BuftabGetIdx(ctx->expr, &buftab_idx, buf_id);
    LG__Assert(status == LG_STATUS_OK); // We should only ever be passed valid buffer ids.
    ctx->expr->buf_table_bytes_required[buftab_idx] = total_size_bytes;

out_release_scratch:
    LG__ReleaseScratch(ctx->scratch, &waypoint);
    return LG_STATUS_OK;
}

void LG_IR__DecorateNodes(struct lg_ir_compilation_context *ctx) {
    enum lg_status status;

    for (size_t i_node = 0; i_node < ctx->expr->nodes_len; i_node++) {
        const uint32_t symbol_ids[3] = {
            ctx->expr->nodes[i_node].y_logical.id,
            ctx->expr->nodes[i_node].x0_logical.id,
            ctx->expr->nodes[i_node].x1_logical.id,
        };
        struct lg_desc *const desc_ptrs[3] = {
            &ctx->expr->nodes[i_node].y_physical,
            &ctx->expr->nodes[i_node].x0_physical,
            &ctx->expr->nodes[i_node].x1_physical,
        };
        size_t *const offset_ptrs[3] = {
            &ctx->expr->nodes[i_node].y_offset,
            &ctx->expr->nodes[i_node].x0_offset,
            &ctx->expr->nodes[i_node].x1_offset,
        };
        size_t *const bufid_ptrs[3] = {
            &ctx->expr->nodes[i_node].y_offset,
            &ctx->expr->nodes[i_node].x0_offset,
            &ctx->expr->nodes[i_node].x1_offset,
        };

        for (size_t i_sym = 0; i_sym < 3; i_sym++) {
            size_t symtab_array_idx;
            bool was_occupied;
            status = LG_IR_SymtabUpsert(&ctx->symtab, &symtab_array_idx, &was_occupied, symbol_ids[i_sym]);
            LG__Assert(status == LG_STATUS_OK);
            LG__Assert(was_occupied);

            *(desc_ptrs[i_sym]) = ctx->symtab.descs[symtab_array_idx];
            *(offset_ptrs[i_sym]) = ctx->symtab.buffer_offsets[symtab_array_idx];
            *(bufid_ptrs[i_sym]) = ctx->symtab.buffer_ids[symtab_array_idx];
        }
    }
}

void LG_IR__DecorateWithMaps(struct lg_ir_compilation_context *ctx) {
    for (size_t i = 0; i < ctx->expr->nodes_len; i++) {
        switch (ctx->expr->nodes[i].opcode) {
        case LG_OPCODE_SOURCE:
        case LG_OPCODE_SINK:
        case LG_OPCODE_NOP:
            continue;
        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB: {
            enum lg_status status = LG_CreateBroadcastSpace((struct lg_desc*[]){
                &ctx->expr->nodes[i].y_physical,         
                &ctx->expr->nodes[i].x0_physical,         
                &ctx->expr->nodes[i].x1_physical,         
            }, 3);
            LG__Assert(status == LG_STATUS_OK);
            break;
        }

        case LG_OPCODE_CONTRACT: {
            enum lg_status status = LG_CreateContractionSpace(
                &ctx->expr->nodes[i].y_physical,
                &ctx->expr->nodes[i].x0_physical,         
                &ctx->expr->nodes[i].x1_physical,
                ctx->expr->nodes[i].meta_as.contract.n_batch_axes
            );
            LG__Assert(status == LG_STATUS_OK);
            break;
        }

        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
            LG__Unreachable("TODO");
        }
    }
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
    struct lg_ir_compilation_context *ctx,
    size_t mem_align
) {
    enum lg_status status;
    status = LG_IR__ValidateExprStructure(ctx);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__InferDims(ctx);
    if (status != LG_STATUS_OK) {
        return status;
    }
    LG_IR__AssignLayouts(ctx, LG_LAYOUT_ROW_MAJOR /* TODO */, mem_align);
    for (size_t i = 0; i <ctx->expr->buf_table_len; i++) {
        const uint32_t buf_id = ctx->expr->buf_table_ids[i];
        status = LG_IR__Bufferize(ctx, buf_id, mem_align);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    // TODO: these functions really need better names
    LG_IR__DecorateNodes(ctx);
    LG_IR__DecorateWithMaps(ctx);
    // status = LG_IR__SortAxes(expr);
    // if (status != LG_STATUS_OK) {
    //     return status;
    // }
    // status = LG_IR__CoalesceAxes(expr);
    // if (status != LG_STATUS_OK) {
    //     return status;
    // }
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
