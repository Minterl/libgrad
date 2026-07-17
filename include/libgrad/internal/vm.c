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
    status = LG_DescComputeStrides(&y->desc, LG_LAYOUT_ROW_MAJOR, 1 /* TODO */);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_ADD,   
        .y = *y,
        .x0 = x0,
        .x1 = x1,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = LG_CreateBroadcastSpace((struct lg_desc*[]){
        &expr->nodes[expr_idx].y.desc,
        &expr->nodes[expr_idx].x0.desc,
        &expr->nodes[expr_idx].x1.desc,
    }, 3);

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
    LG_ComputeContractedDims(&y->desc, x0.desc, x1.desc, n_batch_axes);
    status = LG_DescComputeStrides(&y->desc, LG_LAYOUT_ROW_MAJOR, 1 /* TODO */);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
        .opcode = LG_OPCODE_CONTRACT,   
        .y = *y,
        .x0 = x0,
        .x1 = x1,
    });
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = LG_CreateContractionSpace(
        &expr->nodes[expr_idx].y.desc,
        &expr->nodes[expr_idx].x0.desc,
        &expr->nodes[expr_idx].x1.desc,
        n_batch_axes
    );
    if (status != LG_STATUS_OK) {
        return status;
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

enum lg_status LG_IR_CompileExpr(struct lg_ir_expr *expr) {
    enum lg_status status;
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
