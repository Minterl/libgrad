#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/core.h>

lg_status lg_cpu_backward(lg_tape tape);

/// Tensors must be sorted & broadcasted
lg_status lg_cpu_add(lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_cpu_add_back(const lg_tensor upstream, lg_tensor operand_a, lg_tensor operand_b);

static inline bool __lg_increment_coords_rtl(lg_size *coords, const lg_size *dim, lg_size rank) {
    if (rank == 0) return false;

    lg_size axis = rank;
    
    while (axis > 0) {
        axis--;
        coords[axis]++;
        if (coords[axis] < dim[axis]) {
            return true; 
        }
        coords[axis] = 0;
    }

    return false;
}

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

lg_status lg_cpu_forward(lg_tape tape) {
    for (lg_size i = 0; i < tape.len; i++) {
        switch (tape.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add(
                tape.outputs[i],
                tape.inputs_a[i],
                tape.inputs_b[i]
            );
            break;
        case LG_OPCODE_SUB:
        case LG_OPCODE_CONTRACT:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        case __LG_N_VIRTUAL_OPS: // TODO
        case LG_OPCODE_TRANSPOSE:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    }

    return LG_STATUS_OK;
}

lg_status lg_cpu_backward(lg_tape tape) {
    if (tape.len == 0) {
        return LG_STATUS_OK;
    }
    
    lg_size i = tape.len - 1;
    do {
        switch (tape.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add_back(tape.outputs[i], tape.inputs_a[i], tape.inputs_b[i]);
            break;
        case LG_OPCODE_SUB:
        case LG_OPCODE_CONTRACT:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        case __LG_N_VIRTUAL_OPS: // TODO
        case LG_OPCODE_TRANSPOSE:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    } while (i-- > 0);

    return LG_STATUS_OK;
}

lg_status lg_cpu_add(
    lg_tensor out,
    const lg_tensor a,
    const lg_tensor b
) {
    lg_size coords[LG_MAX_RANK] = {0};

    do {
        lg_size out_idx = 0;
        lg_size a_idx = 0;
        lg_size b_idx = 0;

        for (lg_size i = 0; i < out.rank; i++) {
            out_idx += out.strides[i] * coords[i];
            a_idx += a.strides[i] * coords[i];
            b_idx += b.strides[i] * coords[i];
        }

        out.data[out_idx] = a.data[a_idx] + b.data[b_idx];
    } while (__lg_increment_coords_rtl(coords, out.dim, out.rank));

    return LG_STATUS_OK;
}

lg_status lg_cpu_add_back(const lg_tensor upstream, lg_tensor operand_a, lg_tensor operand_b) {
    lg_size coords[LG_MAX_RANK] = {0};

    do {
        lg_size upstream_idx = 0;
        lg_size a_idx = 0;
        lg_size b_idx = 0;

        for (lg_size i = 0; i < upstream.rank; i++) {
            upstream_idx += upstream.strides[i] * coords[i];
            a_idx += operand_a.strides[i] * coords[i];
            b_idx += operand_b.strides[i] * coords[i];
        }

        operand_a.grad[a_idx] += upstream.grad[upstream_idx];
        operand_b.grad[b_idx] += upstream.grad[upstream_idx];
    } while (__lg_increment_coords_rtl(coords, upstream.dim, upstream.rank));

    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
