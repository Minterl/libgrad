#ifndef LG_CORE_H_
#define LG_CORE_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <libgrad/internal/fnv.h>

/// Maximum possible Tensor rank
/// All tensors will have an array of this size to store
/// dims, so  keep this to a minimum.
#ifndef LG_MAX_RANK
#define LG_MAX_RANK 8
#endif // LG_MAX_RANK

/// The number of tensors tracked by `lg_tracker`.
#ifndef LG_N_TRACKED_TENSORS
#define LG_N_TRACKED_TENSORS 4
#endif // LG_N_TRACKED_TENSORS 4

/// Type to back Tensor data
#ifndef lg_dtype
#define lg_dtype float
#endif // lg_dtype

/// Pointer-sized integer
#ifndef lg_size
#define lg_size size_t
#endif // lg_size

/// Bounds checking
#ifdef __cplusplus
    #define LG_CHECK_BOUNDS(x) /* nothing */
    #define LG_CHECK_BOUNDS_NULLABLE(x) /* nothing */
#else
    #if defined(__clang__) && __has_attribute(counted_by)
        #define LG_CHECK_BOUNDS(x) __attribute__((counted_by(x)))
        #define LG_CHECK_BOUNDS_NULLABLE(x) __attribute__((counted_by_or_null(x)))
    #elif defined(__GNUC__) && (__GNUC__ >= 16) // Pointer support introduced in GCC 16
        #define LG_CHECK_BOUNDS(x) __attribute__((counted_by(x)))
        #define LG_CHECK_BOUNDS_NULLABLE(x) __attribute__((counted_by_or_null(x)))
    #else
        #define LG_CHECK_BOUNDS(x) /* nothing */
        #define LG_CHECK_BOUNDS_NULLABLE(x) /* nothing */
    #endif
#endif

typedef enum lg_status {
    LG_STATUS_OK = 0,
    LG_STATUS_INVALID_RANK,
    LG_STATUS_SHAPE_MISMATCH,
    LG_STATUS_STRIDE_MISMATCH,
    LG_STATUS_TAPE_OVERFLOW,
    LG_STATUS_NULL_POINTER,
    LG_STATUS_UNSUPPORTED_OPCODE,
    LG_STATUS_OUT_OF_MEMORY,
    LG_STATUS_OUT_OF_BOUNDS,
    LG_STATUS_HARDWARE_FAULT,
    LG_STATUS_UNEXPECTED_NAN
} lg_status;

typedef enum lg_layout {
    LG_LAYOUT_ROW_MAJOR,
    LG_LAYOUT_COL_MAJOR,
} lg_layout;

/// Define a tensor ID using an eight-character
/// literal.
#define LG_TENSOR_ID_8 LG_HASH_LITERAL_8

/// Represents a single Tensor backed by `data`
/// and `grad`.
///
/// Generally, these are thin handles that should 
/// live on the stack and be shallow copied as necessary.
///
/// The size `data` and `grad`, respectively, should product of the
/// non-zero values in `dim`.
///
/// Backing buffers are stored stride-major.
typedef struct lg_tensor {
    /// The rank of the tensor.
    /// Must be less than LG_MAX_RANK.
    lg_size rank;

    /// The dimensionality of the tensor.
    lg_size dim[LG_MAX_RANK];

    /// The strides of the tensor.
    /// The order of this array must match that of `dim`.
    lg_size strides[LG_MAX_RANK];
    
    /// The tensor's primary backing buffer.
    lg_dtype *data;
    
    /// The tensors intermediate gradient values during backprop.
    /// If this value is NULL, then no gradients are tracked.
    /// This is the means by which a tensor can be used purely for
    /// inference.
    lg_dtype *grad;
} lg_tensor;

/// Discriminator for an operation.
///
/// The integer representations of opcodes are not designed
/// to be stable and should not be serialized.
typedef enum lg_opcode {
    /// --- BINARY OPERATIONS ---
    /// Element-wise tensor addition
    LG_OPCODE_ADD,
    /// Element-wise tensor subtraction
    LG_OPCODE_SUB,
    /// Generalized tensor contraction i.e
    /// dot-product over strided dimensions.
    /// Is generalizable to N-rank tensors.
    LG_OPCODE_CONTRACT,
    /// Hadamard product
    LG_OPCODE_HADAMARD,
    /// Mean Squared Error loss
    LG_OPCODE_LOSS_MSE,
    /// Cross-entropy loss
    LG_OPCODE_LOSS_CROSS_ENTROPY,

    /// --- UNARY OPERATIONS ---
    /// Element-wise ReLU
    LG_OPCODE_RELU,
    /// Element-wise stable softmax
    LG_OPCODE_STABLE_SOFTMAX,
    /// Element-wise sigmoid
    LG_OPCODE_SIGMOID,
    /// Element-wise natural log
    LG_OPCODE_LN,
} lg_opcode;

/// Tracks the coordinates of LG_N_TRACKED_TENSORS tensors.
///
/// All tensors in a single tracker must be both broadcasted and
/// have their dims sorted in descending order.
typedef struct lg_tracker {
    lg_size coords[LG_MAX_RANK];
    lg_tensor tensors[LG_N_TRACKED_TENSORS];
    lg_size indices[LG_N_TRACKED_TENSORS];
    lg_size n_tracked_dims;
} lg_tracker;

/// Increment the coordinate `axis` on `tracker` and update offsets.
/// 
/// Does not perform any bounds checking.
bool lg_tracker_increment(lg_tracker *tracker, lg_size axis);

/// Recomputes the indices in `tracker` according to its `coords`.
///
/// If you want to "jump" to a specific coordinate in a tensor, this is the
/// easiest way to do it.
void lg_tracker_goto(lg_tracker *tracker, lg_size *coords);

/// A tape used to record operations
///
/// A tape is represented in memory as a structure of arrays.
///
/// If input_b.data == NULL, then the node represents a unary
/// operation.
/// Otherwise, it represents a binary operation as expected.
typedef struct lg_tape {
    lg_size cap;
    lg_size len;
    
    lg_opcode *opcodes  LG_CHECK_BOUNDS(len);
    lg_tensor *inputs_a LG_CHECK_BOUNDS(len);
    lg_tensor *inputs_b LG_CHECK_BOUNDS(len);
    lg_tensor *outputs  LG_CHECK_BOUNDS(len);
} lg_tape;

/// Broadcast tensor views.
///
/// The first tensor in the set is considered the 
/// "primary."
/// The primary tensor is the one that dictates the optimized plan
/// plan for the other tensors. In that, this is the tensor where it is 
/// guaranteed that the contiguous dimension (the dimension with the unit stride)
/// will be accessed sequentially in memory.
lg_status lg_tensor_broadcast(lg_tensor **tensors, lg_size n_tensors);

/// Contracts the dimensions of `out`, inferring the contracted dimensions.
///
/// The contracted dimensions must be aligned at the beginning of `a`, and `b` with batch dimensions
/// following.
lg_status lg_tensor_compute_contracted_dims(lg_tensor *out, lg_tensor *a, lg_tensor *b, lg_size n_batch_axes);

/// Sort dims such that the primary is unit stride first.
/// Follows the same "primary" rule as `lg_tensor_broadcast`.
///
/// Inputs to this function MUST be broadcasted.
lg_status lg_tensor_sort_dims(lg_tensor **tensors, lg_size n_tensors);

/// Coalesce tensor dims to be as flat as possible.
/// Follows the same "primary" rule as `lg_tensor_broadcast`.
///
/// Inputs to this function MUST be broadcasted AND sorted from least to greatest
/// using `lg_tensor_sort_dims`.
lg_status lg_tensor_coalesce_dims(lg_tensor **tensors, lg_size n_tensors);

/// Compute the size in bytes of a tensor's data buffer.
static inline lg_size lg_tensor_size_bytes(const lg_tensor tensor);

/// Copy a vector value to the dim `copy_to_dim`.
/// If `grad` is true, copies to the grad buffer instead of the data buffer.
void lg_tensor_copy_vector(lg_tensor tensor, const lg_dtype *vector, lg_size copy_to_dim, bool grad);

/// Lays out a tensor with pre-populated `dim` and `rank` with the strides to be stored in
/// the order in `layout`. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `unit_align` if `unit_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
lg_status lg_tensor_layout(lg_tensor *tensor, lg_layout layout, lg_size unit_align);

/// Returns true if a tensor is isotropic.
/// 
/// Tensors with a rank of zero, and all scalars are considered isotropic,
/// while all vectors are considered anisotropic.
static inline bool lg_tensor_is_isotropic(const lg_tensor tensor);

lg_status lg_add(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_sub(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_contract(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_hadamard(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_status lg_loss_mse(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_loss_cross_entropy(lg_tape *tape, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_status lg_relu(lg_tape *tape, lg_tensor out, const lg_tensor in);
lg_status lg_stable_softmax(lg_tape *tape, const lg_tensor out, const lg_tensor in);
lg_status lg_sigmoid(lg_tape *tape, lg_tensor out, const lg_tensor in);
lg_status lg_ln(lg_tape *tape, lg_tensor out, const lg_tensor in);

lg_status lg_backward(lg_tape *tape);
lg_tensor lg_transpose(lg_tape *tape, const lg_tensor in);

#endif // LG_CORE_H_


#ifdef LG_CORE_IMPLEMENTATION
#undef LG_CORE_IMPLEMENTATION

#include <libgrad/internal/debug.h>

static inline lg_size lg_tensor_size_bytes(const lg_tensor tensor) {
    if (tensor.rank == 0) {
        return 0;
    }

    lg_size max_offset = 0;
    for (lg_size i = 0; i < tensor.rank; i++) {
        if (tensor.dim[i] > 0) {
            max_offset += (tensor.dim[i] - 1) * tensor.strides[i];
        }
    }

    return (max_offset + 1) * sizeof(lg_dtype);
}

void lg_tensor_copy_vector(lg_tensor tensor, const lg_dtype *vector, lg_size copy_to_dim, bool grad) {
    lg_size dim_offset = 0;
    for (lg_size i = 0; i < copy_to_dim; i++) {
        dim_offset += tensor.dim[i];
    }
    const lg_size n_values = tensor.dim[copy_to_dim];
    lg_dtype *const restrict dest = grad ? tensor.grad : tensor.data;
    for (lg_size i = 0; i < n_values; i++) {
        dest[dim_offset + i] = vector[i];
    }
}

lg_status lg_tensor_layout(lg_tensor *tensor, lg_layout layout, lg_size unit_align) {
#ifdef LG_SAFE
    if (tensor->rank > LG_MAX_RANK) {
        return LG_STATUS_INVALID_RANK;
    }
#endif // LG_SAFE
    lg_size last_stride = 1;
    for (lg_size i = 1; i <= tensor->rank; i++) {
        lg_size axis = layout == LG_LAYOUT_ROW_MAJOR ? tensor->rank - i : i - 1;
        tensor->strides[axis] = last_stride;
        last_stride *= tensor->dim[tensor->rank - i];
        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (unit_align > 1 && i == 1) {
            last_stride = (last_stride + unit_align - 1) & ~(unit_align - 1);
        }
    }

    return LG_STATUS_OK;
}

static inline bool lg_tensor_is_isotropic(const lg_tensor tensor) {
    switch (tensor.rank) {
    // All vectors are anisotropic.
    case 2:
        return 0;
    // All scalars and tensors of rank zero are isotropic.
    case 1:
    case 0:
        return 1;
    default: {
        lg_size last_dim = tensor.dim[0];
        for (lg_size i = 0; i < tensor.rank; i++) {
            if (last_dim != tensor.dim[i]) {
                return 0;
            }
        }
        return 1;
    }
    }
}

void lg_tensor_tranpose(lg_tensor *tensor) {
    for (lg_size i = 0; i < tensor->rank / 2; i++) {
        const lg_size opp = tensor->rank - 1 - i;
        lg_size temp = tensor->dim[i];
        tensor->dim[i] = tensor->dim[opp];
        tensor->dim[opp] = temp;
    }
}

/// Returns the maximum rank of all of the tensors
static inline lg_size __lg_tensor_left_pad_dims(lg_tensor **tensors, lg_size n_tensors) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_tensors; i++) {
        if (tensors[i]->rank > max_rank) {
            max_rank = tensors[i]->rank;
        }
    }

    for (lg_size i = 0; i < n_tensors; i++) {
        if (tensors[i]->rank < max_rank) {
            lg_size shift = max_rank - tensors[i]->rank;
            for (lg_size j = tensors[i]->rank; j > 0; j--) {
                lg_size src_idx = j - 1;
                tensors[i]->dim[src_idx + shift] = tensors[i]->dim[src_idx];
                tensors[i]->strides[src_idx + shift] = tensors[i]->strides[src_idx];
            }
            for (lg_size j = 0; j < shift; j++) {
                tensors[i]->dim[j] = 1;
                tensors[i]->strides[j] = 0;
            }
            tensors[i]->rank = max_rank;
        }
    }
    
    return max_rank;
}
    
lg_status lg_tensor_broadcast(lg_tensor **tensors, lg_size n_tensors) {
    const lg_size max_rank = __lg_tensor_left_pad_dims(tensors, n_tensors);

    // --- Validate that all tensors are broadcast-compatible ---
    // Every tensor participating in the tracking must be broadcast-compatible
    // with every other tensor.
    // Naively, we would perform an O(n^2) check across a matrix of all of the participating tensors.
    // The shortcut below is equivalent.
    //
    // Tensors are broadcast-compatible if all of their dimensions are broadcast-compatible.
    // Two dimensions are broadcast-compatible if one of three things is true:
    // 1) The dimensions are the same.
    // 2) One of the dimensions is 1.
    // 3) One of the dimensions does not exist.
    lg_size master_dim[LG_MAX_RANK];
    for (lg_size j = 0; j < max_rank; j++) {
        master_dim[j] = 1;
    }

    for (lg_size i = 0; i < n_tensors; i++) {
        for (lg_size j = 0; j < max_rank; j++) {
            lg_size dim_current = (j < tensors[i]->rank) ? tensors[i]->dim[j] : 1;
            if (dim_current != 1) {
                if (master_dim[j] == 1) {
                    master_dim[j] = dim_current;
                } 
                else if (master_dim[j] != dim_current) {
                    return LG_STATUS_SHAPE_MISMATCH;
                }
            }
        }
    }
    
    // Since we know all of the tensors are broadcast-compatible, and their
    // dims/strides are all in the same order, we can pre-bake broadcasting into the
    // strides in the views.
    //
    // It's important to remember that both the max_rank array and the dims in the views
    // are right-aligned at this point.
    //
    // This is done in two parts for every tensor along each axis:
    // 1) Setting all strides less than the master to zero, causing the actual
    //    offsets to never move.
    // 2) Aligning the logical dimensions of the tensor with the master.
    for (lg_size i = 0; i < n_tensors; i++) {
        for (lg_size j = 1; j <= max_rank; j++) {
            if (tensors[i]->dim[max_rank - j] < master_dim[max_rank - j]) {
                tensors[i]->strides[max_rank - j] = 0;
                tensors[i]->dim[max_rank - j] = master_dim[max_rank - j];
            }
        }
    }

    // The resulting state of our tensor views looks like this:
    // - All tensors have the same logical rank
    // - All tensors have the exact same logical dims
    // - The only thing that changes between tensor views is
    //   striding.

    return LG_STATUS_OK;
}

lg_status lg_tensor_compute_contracted_dims(lg_tensor *out, lg_tensor *a, lg_tensor *b, lg_size n_batch_axes) {
    // The logical tensor axes will be laid out as follows:
    // { [batch], [a_free], [b_free], [contracted] }
    //    reg      reg       reg       0          | out strides
    //    reg      reg       0         reg        | a strides
    //    reg      0         reg       reg        | b strides

    lg_tensor out_cpy = *out;
    lg_tensor a_cpy = *a;
    lg_tensor b_cpy = *b;
    
    // a.rank = n_batch + n_contracted + a_free
    // b.rank = n_batch + n_contracted + b_free
    // out.rank = n_batch + a_free + b_free
    // ergo ...
    const lg_size n_contracted_axes = (a->rank + b->rank - out->rank - n_batch_axes) / 2;
    const lg_size a_first_contracted_axis = a->rank - n_contracted_axes;
    const lg_size b_first_free_axis = n_contracted_axes + n_batch_axes;

    // Batch axes are already in place
    lg_size r = n_batch_axes;

    // Free axes
    for (lg_size i = n_batch_axes; i < a_first_contracted_axis; i++, r++) {
        a->dim[r] = a_cpy.dim[i];
        a->strides[r] = a_cpy.strides[i];
        b->dim[r] = a_cpy.dim[i];
        b->strides[r] = 0;
        if (out->dim[r] != a_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        out->strides[r] = out_cpy.strides[r];
    }
    for (lg_size i = b_first_free_axis; i < b_cpy.rank; i++, r++) {
        a->dim[r] = b_cpy.dim[i];
        a->strides[r] = 0;
        b->dim[r] = b_cpy.dim[i];
        b->strides[r] = b_cpy.strides[i];
        if (out->dim[r] != b_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        out->strides[r] = out_cpy.strides[r];
    }

    // Contracted axes
    if (n_contracted_axes > 0) {
        for (
            lg_size a_ax = a_first_contracted_axis, b_ax = b_first_free_axis - 1;
            a_ax < a_cpy.rank; // b_ax > 0
            a_ax++, b_ax--, r++
        ) {
            a->dim[r] = a_cpy.dim[a_ax];
            a->strides[r] = a_cpy.strides[a_ax];
            b->dim[r] = a_cpy.dim[a_ax];
            b->strides[r] = b_cpy.strides[b_ax];
            out->dim[r] = a_cpy.dim[a_ax];
            out->strides[r] = 0;
        }
    }

    out->rank = r;
    a->rank = r;
    b->rank = r;

    return LG_STATUS_OK;
}

lg_status lg_tensor_sort_dims(lg_tensor **tensors, lg_size n_tensors) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_tensors; i++) {
        if (tensors[i]->rank > max_rank) {
            max_rank = tensors[i]->rank;
        }
    }

    for (lg_size i = 0; i < max_rank; i++) {
        bool swapped = 0;
        for (lg_size j = 1; j < max_rank - i; j++) {
            const lg_size prev_dim = tensors[0]->strides[j - 1];
            const lg_size cur_dim = tensors[0]->strides[j];
            if (prev_dim < cur_dim) {
                // Swap the dimensions and strides for all of the tensors
                for (lg_size k = 0; k < n_tensors; k++) {
                    lg_size temp = tensors[k]->dim[j - 1];
                    tensors[k]->dim[j - 1] = tensors[k]->dim[j];
                    tensors[k]->dim[j] = temp;
                    temp = tensors[k]->strides[j - 1];
                    tensors[k]->strides[j - 1] = tensors[k]->strides[j];
                    tensors[k]->strides[j] = temp;
                }
                swapped = 1;
            }
        }
        if (!swapped) {
            break;
        }
    }

    return LG_STATUS_OK;
}

lg_status lg_tensor_coalesce_dims(lg_tensor **tensors, lg_size n_tensors) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_tensors; i++) {
        if (tensors[i]->rank > max_rank) {
            max_rank = tensors[i]->rank;
        }
    }

    // --- Coalesce dimensions ---
    // Dimensions can be coalesced under three conditions:
    // 1) One of the dimension is broadcasted
    // 2) One of the dimensions has the unit stride
    //    (is already contiguous in memory)
    // 3) The latter dimensions is the contiguous extension of the 
    //    previous dimenison.
    //
    // Use i + 1 < max_rank to handle underflows.
    for (lg_size i = 0; i + 1 < max_rank; i++) {
        while (i + 1 < max_rank) {
            bool can_coalesce = 1;
            for (lg_size j = 0; j < n_tensors; j++) {
                const lg_size d0 = tensors[j]->dim[i];
                const lg_size d1 = tensors[j]->dim[i + 1];
                const lg_size s0 = tensors[j]->strides[i];
                const lg_size s1 = tensors[j]->strides[i + 1];

                const bool is_broadcasted = s0 == 0 || s1 == 0;
                const bool has_unit = d0 == 1 || d1 == 1;
                const bool is_contiguous_extension = s0 == s1 * d1;

                if (!is_broadcasted && !has_unit && !is_contiguous_extension) {
                    can_coalesce = 0;
                    break;
                }
            }

            if (!can_coalesce) {
                break;
            }

            const lg_size new_dim = tensors[0]->dim[i] * tensors[0]->dim[i + 1];
            max_rank--;
            for (lg_size j = 0; j < n_tensors; j++) {
                tensors[j]->rank = max_rank;
                tensors[j]->dim[i] = new_dim;
                if (tensors[j]->strides[i + 1] != 0) {
                    tensors[j]->strides[i] = tensors[j]->strides[i + 1];
                }
                for (lg_size k = i + 1; k < max_rank; k++) {
                    tensors[j]->dim[k] = tensors[j]->dim[k + 1];
                    tensors[j]->strides[k] = tensors[j]->strides[k + 1];
                }
            }
        }
    }

    return LG_STATUS_OK;
}

static inline lg_status lg_tape_push(
    lg_tape *tape,
    const lg_opcode opcode,
    const lg_tensor a,
    const lg_tensor b,
    const lg_tensor output
) {
#ifdef LG_SAFE
    if (tape->len >= tape->cap) {
        return LG_STATUS_TAPE_OVERFLOW;
    }
#endif // LG_SAFE

    lg_size next_idx = tape->len;
    tape->len += 1;
    tape->opcodes[next_idx] = opcode;
    tape->inputs_a[next_idx] = a;
    tape->inputs_b [next_idx] = b;
    tape->outputs[next_idx] = output;

    return LG_STATUS_OK;
}

bool lg_tracker_increment(lg_tracker *tracker, lg_size axis) {
    const lg_size rank = tracker->tensors[0].rank;
    const lg_size first_tracked_dim = rank - tracker->n_tracked_dims;
    const lg_size *restrict dim = tracker->tensors[0].dim;

    if (rank == 0) {
        return false;
    }

    axis += 1;
    while (axis > first_tracked_dim) {
        axis--;
        tracker->coords[axis]++;
        if (tracker->coords[axis] < dim[axis]) {
            for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
                tracker->indices[i] += tracker->tensors[i].strides[axis];
            }
            return true; 
        }
        tracker->coords[axis] = 0;
        for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
            tracker->indices[i] -= tracker->tensors[i].strides[axis] * (dim[axis] - 1);
        }
    }

    return false;
}

void lg_tracker_goto(lg_tracker *tracker, lg_size *coords) {
    for(lg_size i = 0; i < tracker->n_tracked_dims; i++) {
        tracker->coords[i] = coords[i];
    }

    for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
        tracker->indices[i] = 0;
        for (lg_size j = 0; j < tracker->n_tracked_dims; j++) {
            tracker->indices[i] += tracker->tensors[i].strides[j] * coords[j];
        }
    }
}

lg_status lg_add(
    lg_tape *tape,
    lg_tensor out,
    const lg_tensor a,
    const lg_tensor b
) {
    lg_status status;
    status = lg_tape_push(tape, LG_OPCODE_ADD, a, b, out);
    if (status != LG_STATUS_OK) {
        return status;
    }
    const lg_size i = tape->len - 1;
    status = lg_tensor_broadcast((lg_tensor*[]){
        &tape->outputs[i],
        &tape->inputs_a[i],
        &tape->inputs_b[i],
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = lg_tensor_sort_dims((lg_tensor*[]){
        &tape->outputs[i],
        &tape->inputs_a[i],
        &tape->inputs_b[i],
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = lg_tensor_coalesce_dims((lg_tensor*[]){
        &tape->outputs[i],
        &tape->inputs_a[i],
        &tape->inputs_b[i],
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    return LG_STATUS_OK;
}

#endif // LG_CORE_IMPLEMENTATION
