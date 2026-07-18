#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

struct lg_ir_symbol LG_IR__CreateSymbol(struct lg_ir_expr *expr) {
    const size_t id = expr->next_symbol_id;
    expr->next_symbol_id++;
    struct lg_ir_symbol s = {
        .id = id,
    };
    return s;
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

enum lg_status LG_IR_AppendNop(struct lg_ir_expr *expr, struct lg_ir_symbol x) {
    return LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
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
    enum lg_status status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
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
    enum lg_status status = LG_IR__ExprAppend(expr, (struct lg_ir_expr_node){
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

enum lg_status LG_IR__SortAxes(struct lg_ir_expr *expr) {
    enum lg_status status;
    for (size_t i = 0; i < expr->len; i++) {
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
    for (size_t i = 0; i < expr->len; i++) {
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
