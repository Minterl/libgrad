#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

enum lg_status LG_IR_GetLastLocationOfTensor(struct lg_ir_expr *expr, struct lg_ir_tensor *x) {
    bool found = false;
    for (size_t i = 0; i < expr->len; i++) {
        if (expr->nodes[i].y.born_at == x->born_at) {
            // We only change the data pointer because the expr may have
            // some other `desc` that doesn't match that of the caller.
            // We do not want to mutate the caller's interpretation of the buffer.
            x->data = expr->nodes[i].y.data; 
            found = true;
        }
    }
    if (!found) {
        return LG_STATUS_NOT_FOUND;
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR__ExprAppend(
    struct lg_ir_expr *expr,
    const struct lg_ir_expr_node node 
) {
#ifdef LG_SAFE
    if (expr->len >= expr->cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
#endif // LG_SAFE

    size_t next_idx = expr->len;
    expr->len += 1;
    expr->nodes[next_idx] = node;

    return LG_STATUS_OK;
}

/// Copies the dims and rank from `src` to `dest`.
static inline void LG_IR__CloneLogicalShape(struct lg_ir_tensor *restrict dest, const struct lg_ir_tensor *restrict src) {
    dest->born_at = src->born_at;
    for (size_t i = 0; i < src->desc.rank; i++) {
        dest->desc.dim[i] = src->desc.dim[i];
    }
    dest->desc.rank = src->desc.rank;
}

enum lg_status LG_IR_AppendNop(struct lg_ir_expr *expr, struct lg_ir_tensor x) {
    return LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_NOP,
        .y = (struct lg_ir_tensor){ .born_at = expr->len },
        .x0 = x,
    });
}

enum lg_status LG_IR_AppendAdd(
    struct lg_ir_expr *expr,
    struct lg_ir_tensor *y,
    const struct lg_ir_tensor x0,
    const struct lg_ir_tensor x1
) {
    const size_t expr_idx = expr->len;
    enum lg_status status;

    LG_IR__CloneLogicalShape(y, &x0);
    y->born_at = expr_idx;
    status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_ADD,   
        .y = *y,
        .x0 = x0,
        .x1 = x1,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    return LG_STATUS_OK;
}

enum lg_status LG_IR_AppendContract(
    struct lg_ir_expr *expr,
    struct lg_ir_tensor *y,
    struct lg_ir_tensor x0,
    struct lg_ir_tensor x1,
    const size_t n_batch_axes
) {
    const size_t expr_idx = expr->len;
    enum lg_status status;

    y->born_at = expr_idx;
    size_t rank = 0;
    for (size_t i = x0.desc.rank; i > n_batch_axes; i--, rank++) {
        y->desc.dim[rank] = x0.desc.dim[i - 1];
    }
    for (size_t i = n_batch_axes; i < x1.desc.rank; i++, rank++) {
        y->desc.dim[rank] = x0.desc.dim[i];
    }
    y->desc.rank = rank;

    status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_CONTRACT,   
        .y = *y,
        .x0 = x0,
        .x1 = x1,
        .contract_n_batch_axes = n_batch_axes,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    return LG_STATUS_OK;
}

enum lg_status LG_IR__AssignLayouts(struct lg_ir_expr *expr, enum lg_layout layout, size_t unit_align) {
    for (size_t i_node = 0; i_node < expr->len; i_node++) {
        if (expr->nodes[i_node].opcode == LG_OPCODE_NOP) {
            continue;
        }

        struct lg_desc *const descs[3] = {
            &expr->nodes[i_node].y.desc,
            &expr->nodes[i_node].x0.desc,
            &expr->nodes[i_node].x1.desc,
        };

        for (size_t i_desc = 0; i_desc < 3; i_desc++) {
            struct lg_desc *desc = descs[i_desc];

            for (size_t i_stride = 0; i_stride < LG_MAX_RANK; i_stride++) {
                if (desc->strides[i_stride] != 0) {
                    goto skip_layout;
                }
            }

            enum lg_status status = LG_DescComputeLayoutStrides(desc, layout, unit_align);
            if (status != LG_STATUS_OK) {
                return status;
            }

skip_layout:;
        }
    }

    return LG_STATUS_OK;
}

enum lg_status LG_IR__PrecomputeStrides(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->len; i++) {
        switch (expr->nodes[i].opcode) {
        case LG_OPCODE_NOP:
            break;

        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB:
        case LG_OPCODE_HADAMARD: {
            status = LG_ComputeBroadcastedAxes((struct lg_desc*[]){
                &expr->nodes[i].y.desc,
                &expr->nodes[i].x0.desc,
                &expr->nodes[i].x1.desc,
            }, 3);
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_CONTRACT: {
            status = LG_ComputeContractedAxes(
                &expr->nodes[i].y.desc,
                &expr->nodes[i].x0.desc,
                &expr->nodes[i].x1.desc,
                expr->nodes[i].contract_n_batch_axes
            );
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR__SortAxes(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->len; i++) {
        status = LG_SortAxes((struct lg_desc*[]){
            &expr->nodes[i].y.desc,
            &expr->nodes[i].x0.desc,
            &expr->nodes[i].x1.desc,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR__CoalesceAxes(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->len; i++) {
        status = LG_CoalesceAxes((struct lg_desc*[]){
            &expr->nodes[i].y.desc,
            &expr->nodes[i].x0.desc,
            &expr->nodes[i].x1.desc,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

enum lg_status LG_IR_CompileExpr(struct lg_ir_expr *expr, enum lg_layout layout, size_t unit_align) {
    enum lg_status status;
    status = LG_IR__AssignLayouts(expr, layout, unit_align);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = LG_IR__PrecomputeStrides(expr);
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
