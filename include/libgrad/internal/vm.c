#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

lg_status lg_get_last_location(lg_expr *expr, lg_tensor *x) {
    bool found = false;
    for (lg_size i = 0; i < expr->len; i++) {
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

lg_status __lg_expr_append(
    lg_expr *expr,
    const lg_expr_node node 
) {
#ifdef LG_SAFE
    if (expr->len >= expr->cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
#endif // LG_SAFE

    lg_size next_idx = expr->len;
    expr->len += 1;
    expr->nodes[next_idx] = node;

    return LG_STATUS_OK;
}

/// Copies the dims and rank from `src` to `dest`.
static inline void __lg_tensor_clone_logical_shape(lg_tensor *restrict dest, const lg_tensor *restrict src) {
    dest->born_at = src->born_at;
    for (lg_size i = 0; i < src->desc.rank; i++) {
        dest->desc.dim[i] = src->desc.dim[i];
    }
    dest->desc.rank = src->desc.rank;
}

lg_status lg_nop(lg_expr *expr, lg_tensor x) {
    return __lg_expr_append(expr, (lg_expr_node){
        .opcode = LG_OPCODE_NOP,
        .y = (lg_tensor){ .born_at = expr->len },
        .x0 = x,
    });
}

lg_status lg_add(
    lg_expr *expr,
    lg_tensor *y,
    const lg_tensor x0,
    const lg_tensor x1
) {
    const lg_size expr_idx = expr->len;
    lg_status status;

    __lg_tensor_clone_logical_shape(y, &x0);
    y->born_at = expr_idx;
    status = __lg_expr_append(expr, (lg_expr_node){
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

lg_status lg_contract(
    lg_expr *expr,
    lg_tensor *y,
    lg_tensor x0,
    lg_tensor x1,
    const lg_size n_batch_axes
) {
    const lg_size expr_idx = expr->len;
    lg_status status;

    y->born_at = expr_idx;
    lg_size rank = 0;
    for (lg_size i = x0.desc.rank; i > n_batch_axes; i--, rank++) {
        y->desc.dim[rank] = x0.desc.dim[i - 1];
    }
    for (lg_size i = n_batch_axes; i < x1.desc.rank; i++, rank++) {
        y->desc.dim[rank] = x0.desc.dim[i];
    }
    y->desc.rank = rank;

    status = __lg_expr_append(expr, (lg_expr_node){
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

lg_status __lg_expr_pass_assign_layouts(lg_expr *expr, lg_layout layout, lg_size unit_align) {
    for (lg_size i_node = 0; i_node < expr->len; i_node++) {
        if (expr->nodes[i_node].opcode == LG_OPCODE_NOP) {
            continue;
        }

        lg_desc *const descs[3] = {
            &expr->nodes[i_node].y.desc,
            &expr->nodes[i_node].x0.desc,
            &expr->nodes[i_node].x1.desc,
        };

        for (lg_size i_desc = 0; i_desc < 3; i_desc++) {
            lg_desc *desc = descs[i_desc];

            for (lg_size i_stride = 0; i_stride < LG_MAX_RANK; i_stride++) {
                if (desc->strides[i_stride] != 0) {
                    goto skip_layout;
                }
            }

            lg_status status = lg_desc_layout(desc, layout, unit_align);
            if (status != LG_STATUS_OK) {
                return status;
            }

skip_layout:;
        }
    }

    return LG_STATUS_OK;
}

lg_status __lg_expr_pass_precompute_strides(lg_expr *expr) {
    lg_status status;
    for (lg_size i = 0; i < expr->len; i++) {
        switch (expr->nodes[i].opcode) {
        case LG_OPCODE_NOP:
            break;

        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB:
        case LG_OPCODE_HADAMARD: {
            status = lg_desc_broadcast((lg_desc*[]){
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
            status = lg_desc_compute_contracted_dims(
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

lg_status __lg_expr_pass_sort_axes(lg_expr *expr) {
    lg_status status;
    for (lg_size i = 0; i < expr->len; i++) {
        status = lg_desc_sort_axes((lg_desc*[]){
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

lg_status __lg_expr_pass_coalesce_axes(lg_expr *expr) {
    lg_status status;
    for (lg_size i = 0; i < expr->len; i++) {
        status = lg_desc_coalesce_axes((lg_desc*[]){
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

lg_status lg_expr_compile(lg_expr *expr, lg_layout layout, lg_size unit_align) {
    lg_status status;
    status = __lg_expr_pass_assign_layouts(expr, layout, unit_align);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = __lg_expr_pass_precompute_strides(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = __lg_expr_pass_sort_axes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = __lg_expr_pass_coalesce_axes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    return LG_STATUS_OK;
}
