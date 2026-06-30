#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/core.h>

lg_status lg_cpu_backward(lg_tape tape);

/// Tensors must be sorted & broadcasted
lg_status lg_cpu_add(lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_cpu_add_back(const lg_tensor upstream, lg_tensor operand_a, lg_tensor operand_b);

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
        default:
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
        default:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    } while (i-- > 0);

    return LG_STATUS_OK;
}

lg_status lg_cpu_add(lg_tensor out, const lg_tensor a, const lg_tensor b) {
    lg_tracker tracker = {
        .tensors = {out, a, b},
        .n_tracked_dims = out.rank,
    };

    do {
        const lg_size out_idx = tracker.indices[0];
        const lg_size a_idx = tracker.indices[1];
        const lg_size b_idx = tracker.indices[2];

        out.data[out_idx] = a.data[a_idx] + b.data[b_idx];
   } while (lg_tracker_increment(&tracker, out.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_add_back(const lg_tensor upstream, lg_tensor operand_a, lg_tensor operand_b) {
    lg_tracker tracker = {
        .tensors = {upstream, operand_a, operand_b},
        .n_tracked_dims = upstream.rank,
    };

    do {
        const lg_size upstream_idx = tracker.indices[0];
        const lg_size a_idx = tracker.indices[1];
        const lg_size b_idx = tracker.indices[2];

        operand_a.grad[a_idx] += upstream.grad[upstream_idx];
        operand_b.grad[b_idx] += upstream.grad[upstream_idx];
    } while (lg_tracker_increment(&tracker, upstream.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_contract(lg_tensor out, const lg_tensor a, const lg_tensor b) {
    lg_tracker tracker = {
        .tensors = {out, a, b},
        .n_tracked_dims = out.rank,
    };

    do {
        const lg_size out_idx = tracker.indices[0];
        const lg_size a_idx = tracker.indices[1];
        const lg_size b_idx = tracker.indices[2];

        out.data[out_idx] += a.data[a_idx] * b.data[b_idx];
    } while (lg_tracker_increment(&tracker, out.rank - 1));

    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
