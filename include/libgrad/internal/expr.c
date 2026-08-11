#include <libgrad/internal/expr.h>
#include <libgrad/internal/core.h>
#include <libgrad/internal/debug.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// 
/// error reporting shennanigans
///
/// TODO: move this with the context stuff to a context.(h|c)
///
////////////////////////////////////////////////////////////////////////////////

void
lg_perror(LG_Context *ctx, LG_Writer *writer) {
    lg_str8 as_string = (lg_str8){.len = ctx->err_msg_len, .p = ctx->err_msg_backing_buf};
    lg_printf(writer, as_string);
    lg_printf(writer, lg_str8_lit("\n"));
}

size_t 
lg_report_error_write(void *ctx_, lg_str8 str) {
    LG_Context *ctx = ctx_;
    size_t bytes_written = lg_strcpy((lg_str8){
        .len = LG_MAX_ERR_LEN - ctx->err_msg_len,
        .p = ctx->err_msg_backing_buf + ctx->err_msg_len,
    }, str);
    ctx->err_msg_len += bytes_written;
    return bytes_written;
}

/// Reports error on a best-effort basis, filling the buffer as much as possible.
/// Does nothing if the error has already been set
void 
lg_report_error(LG_Context *ctx, LG_StatusKind status, lg_str8 fmt, ...) {
    if (ctx->last_status != LG_StatusKind_OK) {
        return;
    }

    lg_assert(status != LG_StatusKind_OK);

    ctx->err_msg_len = 0;
    ctx->last_status = status;

    LG_Writer w = {
        .ctx = (void*)ctx,
        .write = lg_report_error_write,
    };

    va_list ap;
    va_start(ap, fmt);
    LG_StatusKind vprintf_status = lg_vprintf(&w, fmt, ap);
    (void)vprintf_status;
    va_end(ap);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// private logical expression building utilities
///
////////////////////////////////////////////////////////////////////////////////

#define lg_builder_append(ctx, builder, opcode, ...) lg_builder_append_((ctx), (builder), (opcode), (LG_BuilderAppendOptions){__VA_ARGS__})

typedef struct
LG_BuilderAppendOptions {
    LG_Symbol x0;
    LG_Symbol x1;

    LG_SymbolFlags y_flags;
    LG_ExprNodeMeta meta_as;
} LG_BuilderAppendOptions;

LG_Symbol
lg_builder_append_(
    LG_Context *ctx,
    LG_Builder *builder,
    LG_Opcode opcode,
    LG_BuilderAppendOptions opts
) {
    LG_BuilderNode *node = lg_arena_alloc_struct(&ctx->arena, LG_BuilderNode);
    if (node == NULL) {
        lg_report_error(ctx, LG_StatusKind_OutOfMemory, lg_str8_lit("ran out of memory appending to logical expr"));
        return lg_nil(LG_Symbol);
    }

    builder->next_symbol_id++; // first valid symbol id is 1
    LG_Symbol y = (LG_Symbol){ .id = builder->next_symbol_id };

    node->opcode = opcode;
    node->x0 = opts.x0;
    node->x1 = opts.x1;
    node->y = y;
    node->y_flags = opts.y_flags;
    node->meta_as = opts.meta_as;

    if (builder->ir_tail != NULL) {
        node->prev = builder->ir_tail;
    }
    builder->ir_tail = node;

    return y;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// public logical expression building utilities
///
////////////////////////////////////////////////////////////////////////////////

LG_Symbol
lg_param(LG_Context *ctx, LG_Builder *builder, LG_LogicalShape shape) {
    return lg_builder_append(ctx, builder, LG_Opcode_Param, .meta_as.param.y_shape = shape);
}

void
lg_pin(LG_Context *ctx, LG_Builder *builder, LG_Symbol sym) {
    LG_BuilderNode *iter_node = builder->ir_tail;
    while (iter_node != NULL) {
        if (iter_node->y.id != sym.id) {
            continue;
        }
        iter_node->y_flags |= LG_SymbolFlag_Pin;
    }

    lg_report_error(ctx, LG_StatusKind_InvalidArgument, 
        lg_str8_lit("did not find symbol with id %{i64} while attempting to pin it"), sym.id
    );
}

LG_Symbol
lg_add(LG_Context *ctx, LG_Builder *builder, LG_Symbol x0, LG_Symbol x1) {
    return lg_builder_append(ctx, builder, LG_Opcode_Add, .x0 = x0, .x1 = x1);
}

LG_Symbol
lg_contract(
    LG_Context *ctx,
    LG_Builder *builder,
    LG_Symbol x0,
    LG_Symbol x1,
    size_t n_contracted_axes,
    size_t n_batch_axes
) {
    return lg_builder_append(
        ctx, builder, LG_Opcode_Contract,
        .x0 = x0,
        .x1 = x1,
        .meta_as.contract.n_contracted_axes = n_contracted_axes,
        .meta_as.contract.n_batch_axes = n_batch_axes
    );
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// freeze a logical expression builder into contiguous memory &
/// perform early validation
///
////////////////////////////////////////////////////////////////////////////////

LG_StatusKind
lg_builder_finish(
    LG_Context *ctx,
    LG_Builder *builder,
    LG_Allocator *artifact_allocator,
    LG_LogicalExpr *out_lexpr
) {
    if (builder->next_symbol_id == 0 || builder->ir_tail == NULL) {
        lg_report_error(ctx, LG_StatusKind_InvalidArgument, lg_str8_lit("attempted to finish empty builder"));
        return LG_StatusKind_InvalidArgument;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // ~~ find the length of the expr & validate two invariants: ~~
    // 1) no symbol id > the current ctx.next_symbol_id
    //    - we do this b/c lets us use raw integer indices for the symbol table
    // 2) there are no cycles in the sll

    size_t lexpr_len = 0;
    const size_t max_symbol_id = builder->next_symbol_id;
    {
        bool is_first_iteration = true;
        LG_BuilderNode *tortoise = builder->ir_tail;
        LG_BuilderNode *hare = builder->ir_tail;
        for (; tortoise != NULL; tortoise = tortoise->prev, lexpr_len++) {
            if (hare != NULL) {
                hare = hare->prev;
            }
            if (hare != NULL) {
                hare = hare->prev;
            }

            if (lg_unlikely(!is_first_iteration && hare != NULL && tortoise == hare)) {
                lg_report_error(ctx, LG_StatusKind_InvalidArgument, lg_str8_lit("detected cycle in expr builder node list"));
                return LG_StatusKind_InvalidArgument;
            }

            if (lg_unlikely(tortoise->y.id >= max_symbol_id)) {
                lg_report_error(ctx, LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("found invalid, discontiguous symbol id %{i64} at expr node %{i64}"), tortoise->y.id, lexpr_len
                );
                return LG_StatusKind_InvalidArgument;
            } else if (lg_unlikely(tortoise->x0.id >= max_symbol_id)) {
                lg_report_error(ctx, LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("found invalid, discontiguous symbol id %{i64} at expr node %{i64}"), tortoise->x0.id, lexpr_len
                );
                return LG_StatusKind_InvalidArgument;
            } else if (lg_unlikely(tortoise->x1.id >= max_symbol_id)) {
                lg_report_error(ctx, LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("found invalid, discontiguous symbol id %{i64} at expr node %{i64}"), tortoise->x1.id, lexpr_len
                );
                return LG_StatusKind_InvalidArgument;
            }

            is_first_iteration = false;
        }
    }


    /////////////////////////////////////////////////
    // ~~ copy the data into a contigous array ~~
    
    LG_LogicalExprNode *lexpr_nodes = (LG_LogicalExprNode*)lg_alloc_zero(artifact_allocator, lexpr_len * sizeof(LG_LogicalExprNode));
    if (lexpr_nodes == NULL) {
        lg_report_error(ctx, LG_StatusKind_OutOfMemory, lg_str8_lit("ran out of memory allocating logical expr nodes"));
        return LG_StatusKind_OutOfMemory;
    }

    LG_BuilderNode *iter_node = builder->ir_tail;
    for (
        size_t i_rev = 0;
        iter_node != NULL;
        iter_node = iter_node->prev, i_rev++
    ) { 
        lg_assert(i_rev < lexpr_len);
        const size_t i = lexpr_len - 1 - i_rev;
        
        lexpr_nodes[i].opcode  = iter_node->opcode;
        lexpr_nodes[i].y       = iter_node->y;
        lexpr_nodes[i].x0      = iter_node->x0;
        lexpr_nodes[i].x1      = iter_node->x1;
        lexpr_nodes[i].y_flags = iter_node->y_flags;
        lexpr_nodes[i].meta_as = iter_node->meta_as;
    }


    ////////////////////////////////////////
    // ~~ fin ~~

    out_lexpr->max_symbol_id = max_symbol_id;
    out_lexpr->len = lexpr_len;
    out_lexpr->nodes = lexpr_nodes;

    return LG_StatusKind_OK;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// bit vector utilities (quite useful for the following passes)
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
LG_BitVector {
    size_t n_blocks;
    uint64_t *blocks lg_check_bounds(n_blocks);
} LG_BitVector;

lg_force_inline LG_StatusKind
lg_bv_init(LG_BitVector *bv, LG_Arena *arena, size_t set_width) {
    lg_memzero(bv, sizeof(LG_BitVector));

    size_t n_blocks = lg_align_up(set_width, 64) / 64;
    uint64_t *blocks = (uint64_t*)lg_arena_alloc(arena, n_blocks * sizeof(uint64_t), 4);
    if (blocks == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    bv->n_blocks = n_blocks;
    bv->blocks = blocks;
}

lg_force_inline void 
lg_bv_set_bit(LG_BitVector *bv, bool bit, size_t offset_rtl) {
    const size_t idx = bv->n_blocks - 1 - (offset_rtl / 64);
    const size_t shift = offset_rtl % 64;

    if (bit) {
        bv->blocks[idx] |= UINT64_C(0x1) << shift;
    } else {
        bv->blocks[idx] &= ~(UINT64_C(0x1) << shift);
    }
}

lg_force_inline bool 
lg_bv_get_bit(const LG_BitVector *bv, size_t offset_rtl) {
    const size_t idx = bv->n_blocks - 1 - (offset_rtl / 64);
    const size_t shift = offset_rtl % 64;

    return (bv->blocks[idx] & (UINT64_C(0x1) << shift)) != 0;
}

lg_force_inline void 
lg_bv_or(LG_BitVector *bv, uint64_t *b) {
    for (size_t i = 0; i < bv->n_blocks; i++) {
        bv->blocks[i] |= b[i];
    }
}

lg_force_inline void 
lg_bv_and(LG_BitVector *bv, uint64_t *b) {
    for (size_t i = 0; i < bv->n_blocks; i++) {
        bv->blocks[i] &= b[i];
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// passes for lowering a logical expression into a physical one
///
////////////////////////////////////////////////////////////////////////////////

/// validate functional structure of the expression
LG_StatusKind
lg_validate_lexpr_structure(LG_Context *ctx, LG_LogicalExpr *lexpr) {
    LG_StatusKind status = LG_StatusKind_OK;
    LG_Scope scope = lg_push_scope(&ctx->arena);

    //////////////////////////////////////////////////////////////
    // ~~ param dominance validation ~~
    // this is very easy b/c there' no control flow to speak of
    
    {
        bool params_begin = false;
        bool params_end = false;
            
        for (size_t i = 0; i < lexpr->len; i++) {
            // Param declarations must be the first section of the lexpr,
            // while sink declarations must be at the end.
            if (lexpr->nodes[i].opcode == LG_Opcode_Param && !params_begin) {
                if (i != 0) {
                    lg_report_error(ctx, LG_StatusKind_InvalidArgument, lg_str8_lit(
                        "found the first param declaration at node index %{i64}\n"
                        "note: param declarations must be the first thing in the expr"
                    ), i);
                    status = LG_StatusKind_InvalidArgument;
                    goto out;
                }
                params_begin = true;
            } else if (lexpr->nodes[i].opcode == LG_Opcode_Param && params_end) {
                lg_report_error(ctx, LG_StatusKind_InvalidArgument, lg_str8_lit(
                    "found a param declaration after a non-param operation at node index %{i64}\n"
                    "note: param declarations must happen one after another"
                ), i);
                status = LG_StatusKind_InvalidArgument;
                goto out;
            } else if (lexpr->nodes[i].opcode != LG_Opcode_Param && params_begin) {
                params_end = true;
            }
        }
    }


    //////////////////////////////////////////////
    // ~~ scope validation ~~

    {
        LG_BitVector seen_set;
        status = lg_bv_init(&seen_set, &ctx->arena, lexpr->max_symbol_id);
        if (status == LG_StatusKind_OutOfMemory) {
            lg_report_error(ctx, LG_StatusKind_OutOfMemory, lg_str8_lit("ran out of memory allocating scratch space for validation"));
            goto out;
        } else if (status != LG_StatusKind_OK) {
            lg_report_error(ctx, status, lg_str8_lit("failed to initialize a scratch structure during validation"));
            goto out;
        }

        for (size_t i = 0; i < lexpr->len; i++) {
            const uint32_t new_id = lexpr->nodes[i].y.id;
            const bool found_x0 = lg_bv_get_bit(&seen_set, lexpr->nodes[i].x0.id);
            const bool found_x1 = lg_bv_get_bit(&seen_set, lexpr->nodes[i].x1.id);
            const bool found_y  = lg_bv_get_bit(&seen_set, new_id);

            if (new_id == 0) {
                lg_report_error(ctx,LG_StatusKind_InvalidArgument, lg_str8_lit(
                    "found a symbol with id 0 at node index %{i64}\n"
                    "0 is a reserved symbol id (invalid)"
                ));
                status = LG_StatusKind_InvalidArgument;
                goto out;
            }

            if (found_y) {
                lg_report_error(
                    ctx, 
                    LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("symbol %{i64} was born for the second time at node index %{i64}, violating SSA"),
                    lexpr->nodes[i].y.id, i
                );
                status = LG_StatusKind_InvalidArgument;
                goto out;
            } else if (!found_x0)  {
                lg_report_error(
                    ctx, 
                    LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("use of unknown symbol with id as operand x0 %{i64} at node index %{i64}"),
                    lexpr->nodes[i].x0.id, i
                );
                status = LG_StatusKind_InvalidArgument;
                goto out;
            } else if (!found_x1 && lg_opcode_is_binary(lexpr->nodes[i].opcode)) {
                lg_report_error(
                    ctx, 
                    LG_StatusKind_InvalidArgument, 
                    lg_str8_lit("use of unknown symbol with id as operand x1 %{i64} at node index %{i64}"),
                    lexpr->nodes[i].x1.id, i
                );
                status = LG_StatusKind_InvalidArgument;
                goto out;
            }

            lg_bv_set_bit(&seen_set, 1, new_id);
        }
    }

out:
    lg_pop_scope(&ctx->arena, scope);
    return status;
}

LG_StatusKind
lg_lower_lexpr(
    LG_Context *ctx,
    LG_LogicalExpr *lexpr,
    LG_LogicalExprLoweringFlags flags,
    LG_PhysicalExpr *out_pexpr
) {
    LG_StatusKind status;

    if (!(flags & LG_LogicalExprLoweringFlag_NoStructuralInvariantValidation)) {
        status = lg_validate_lexpr_structure(ctx, lexpr);
        if (status != LG_StatusKind_OK) {
            return status;
        }
    }
}
