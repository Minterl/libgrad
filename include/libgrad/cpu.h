#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/internal/vm.h>
#include <libgrad/internal/core.h>

enum lg_status lgrt_cpu_ExecExpr(struct lgvm_expr expr);

/// Tensors must be sorted & broadcasted
void lgrt_cpu_Add(
    const struct lg_desc y_desc, lg_scalar *restrict y,
    const struct lg_desc x0_desc, const lg_scalar *restrict x0,
    const struct lg_desc x1_desc, const lg_scalar *restrict x1
);
void lgrt_cpu_Contract(
    const struct lg_desc y_desc, lg_scalar *restrict y,
    const struct lg_desc x0_desc, const lg_scalar *restrict x0,
    const struct lg_desc x1_desc, const lg_scalar *restrict x1
);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

enum lg_status lgrt_cpu_ExecExpr(struct lgvm_expr expr) {
    for (size_t i = 0; i < expr.len; i++) {
        switch (expr.nodes[i].opcode) {
        case LG_OPCODE_NOP:
            break;
        case LG_OPCODE_ADD:
            lgrt_cpu_Add(
                expr.nodes[i].y.desc, expr.nodes[i].y.data,
                expr.nodes[i].x0.desc, expr.nodes[i].x0.data,
                expr.nodes[i].x1.desc, expr.nodes[i].x1.data
            );
            break;
        case LG_OPCODE_CONTRACT:
            lgrt_cpu_Contract(
                expr.nodes[i].y.desc, expr.nodes[i].y.data,
                expr.nodes[i].x0.desc, expr.nodes[i].x0.data,
                expr.nodes[i].x1.desc, expr.nodes[i].x1.data
            );
            break;
        case LG_OPCODE_SUB:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        default:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    }

    return LG_STATUS_OK;
}

void lgrt_cpu_Add(
    const struct lg_desc y_desc, lg_scalar *restrict y,
    const struct lg_desc x0_desc, const lg_scalar *restrict x0,
    const struct lg_desc x1_desc, const lg_scalar *restrict x1
) {
    struct lg_nditer iter = {
        .descs = {y_desc, x0_desc, x1_desc},
        .n_tracked_dims = y_desc.rank,
    };

    do {
        const size_t y_idx = iter.indices[0];
        const size_t x0_idx = iter.indices[1];
        const size_t x1_idx = iter.indices[2];

        y[y_idx] = x0[x0_idx] + x1[x1_idx];
   } while (LG_NDiterIncrement(&iter, y_desc.rank - 1));
}

void lgrt_cpu_Contract(
    const struct lg_desc y_desc, lg_scalar *restrict y,
    const struct lg_desc x0_desc, const lg_scalar *restrict x0,
    const struct lg_desc x1_desc, const lg_scalar *restrict x1
) {
    struct lg_nditer iter = {
        .descs = {y_desc, x0_desc, x1_desc},
        .n_tracked_dims = y_desc.rank,
    };

    do {
        const size_t y_idx = iter.indices[0];
        const size_t x0_idx = iter.indices[1];
        const size_t x1_idx = iter.indices[2];

        y[y_idx] += x0[x0_idx] * x1[x1_idx];
    } while (LG_NDiterIncrement(&iter, y_desc.rank - 1));
}

#endif // LG_CPU_IMPLEMENTATION
