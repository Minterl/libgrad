#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

lg_status lg_get_last_location(lg_expr *expr, lg_tensor *x) {
    bool found = false;
    for (lg_size i = 0; i < expr->len; i++) {
        if (expr->y[i].born_at == x->born_at) {
            // We only change the data pointer because the expr may have
            // some other `desc` that doesn't match that of the caller.
            // We do not want to mutate the caller's interpretation of the buffer.
            x->data = expr->y[i].data; 
            found = true;
        }
    }
    if (!found) {
        return LG_STATUS_NOT_FOUND;
    }
    return LG_STATUS_OK;
}

lg_status lg_expr_append(
    lg_expr *expr,
    const lg_opcode opcode,
    const lg_tensor y,
    const lg_tensor x0,
    const lg_tensor x1
) {
#ifdef LG_SAFE
    if (expr->len >= expr->cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
#endif // LG_SAFE

    lg_size next_idx = expr->len;
    expr->len += 1;
    expr->opcodes[next_idx] = opcode;
    expr->y[next_idx] = y;
    expr->x0[next_idx] = x0;
    expr->x1[next_idx] = x1;

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
    return lg_expr_append(expr, LG_OPCODE_NOP, (lg_tensor){ .born_at = expr->len }, x, (lg_tensor){0});
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
    status = lg_expr_append(expr, LG_OPCODE_ADD, *y, x0, x1);
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

    status = lg_expr_append(expr, LG_OPCODE_CONTRACT, *y, x0, x1);
    if (status != LG_STATUS_OK) {
        return status;
    }

    expr->meta[expr_idx].op_contract.n_batch_axes = n_batch_axes;

    return LG_STATUS_OK;
}

lg_status __lg_expr_pass_assign_layouts(lg_expr *expr, lg_layout layout, lg_size unit_align) {
    for (lg_size i_op = 0; i_op < expr->len; i_op++) {
        if (expr->opcodes[i_op] == LG_OPCODE_NOP) {
            continue;
        }

        lg_desc *const descs[3] = {
            &expr->y[i_op].desc,
            &expr->x0[i_op].desc,
            &expr->x1[i_op].desc,
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
        switch (expr->opcodes[i]) {
        case LG_OPCODE_NOP:
            break;

        case LG_OPCODE_ADD:
        case LG_OPCODE_SUB:
        case LG_OPCODE_HADAMARD: {
            status = lg_desc_broadcast((lg_desc*[]){
                &expr->y->desc,
                &expr->x0[i].desc,
                &expr->x1[i].desc,
            }, 3);
            if (status != LG_STATUS_OK) {
                return status;
            }
            break;
        }

        case LG_OPCODE_CONTRACT: {
            status = lg_desc_compute_contracted_dims(
                &expr->y[i].desc,
                &expr->x0[i].desc,
                &expr->x1[i].desc,
                expr->meta[i].op_contract.n_batch_axes
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
            &expr->y[i].desc,
            &expr->x0[i].desc,
            &expr->x1[i].desc,
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
            &expr->y[i].desc,
            &expr->x0[i].desc,
            &expr->x1[i].desc,
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
