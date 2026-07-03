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

/// The number of tensors tracked by `lg_nditer`.
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
    LG_STATUS_EXPR_OVERFLOW,
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

/// Tensor shape descriptor
typedef struct lg_desc {
    /// The rank of the tensor.
    /// Must be less than LG_MAX_RANK.
    lg_size rank;

    /// The dimensionality of the tensor.
    lg_size dim[LG_MAX_RANK];

    /// The strides of the tensor.
    /// The order of this array must match that of `dim`.
    lg_size strides[LG_MAX_RANK];
} lg_desc;

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
    /// The shape descriptor of the tensor.
    lg_desc desc;
    
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
/// All tensors in a single iter must be both broadcasted and
/// have their dims sorted in descending order.
typedef struct lg_nditer {
    lg_size coords[LG_MAX_RANK];
    lg_desc descs[LG_N_TRACKED_TENSORS];
    lg_size indices[LG_N_TRACKED_TENSORS];
    lg_size n_tracked_dims;
} lg_nditer;

/// Increment the coordinate `axis` on `iter` and update offsets.
/// 
/// Does not perform any bounds checking.
bool lg_nditer_increment(lg_nditer *iter, lg_size axis);

/// Recomputes the indices in `iter` according to its `coords`.
///
/// If you want to "jump" to a specific coordinate in a tensor, this is the
/// easiest way to do it.
void lg_nditer_goto(lg_nditer *iter, lg_size *coords);

/// An expression graph used to record operations
///
/// An expression is represented in memory as a structure of arrays.
///
/// If x1.data == NULL, then the node represents a unary
/// operation.
/// Otherwise, it represents a binary operation as expected.
typedef struct lg_expr {
    lg_size cap;
    lg_size len;
    
    lg_opcode *opcodes  LG_CHECK_BOUNDS(len);
    lg_tensor *y        LG_CHECK_BOUNDS(len);
    lg_tensor *x0       LG_CHECK_BOUNDS(len);
    lg_tensor *x1       LG_CHECK_BOUNDS(len);
} lg_expr;

/// Broadcast tensor views.
///
/// The first tensor in the set is considered the 
/// "primary."
/// The primary tensor is the one that dictates the optimized plan
/// plan for the other tensors. In that, this is the tensor where it is 
/// guaranteed that the contiguous dimension (the dimension with the unit stride)
/// will be accessed sequentially in memory.
lg_status lg_desc_broadcast(lg_desc **descs, lg_size n_descs);

/// Contracts the dimensions of `y`, inferring the contracted dimensions.
///
/// The contracted dimensions must be aligned at the beginning of `x0`, and `x1` with batch dimensions
/// following.
lg_status lg_desc_compute_contracted_dims(lg_desc *y, lg_desc *x0, lg_desc *x1, lg_size n_batch_axes);

/// Sort dims such that the primary is unit stride first.
/// Follows the same "primary" rule as `lg_desc_broadcast`.
///
/// Inputs to this function MUST be broadcasted.
lg_status lg_desc_sort_dims(lg_desc **descs, lg_size n_descs);

/// Coalesce tensor dims to be as flat as possible.
/// Follows the same "primary" rule as `lg_desc_broadcast`.
///
/// Inputs to this function MUST be broadcasted AND sorted from least to greatest
/// using `lg_desc_sort_dims`.
lg_status lg_desc_coalesce_dims(lg_desc **descs, lg_size n_descs);

/// Compute the size in bytes of a tensor's data buffer.
static inline lg_size lg_desc_size_bytes(lg_desc desc);

/// Copy a vector value to the dim `copy_to_dim`.
void lg_copy_vector(lg_desc desc, lg_dtype *restrict dest, const lg_dtype *vector, lg_size copy_to_dim);

/// Lays out a tensor with pre-populated `dim` and `rank` with the strides to be stored in
/// the order in `layout`. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `unit_align` if `unit_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
lg_status lg_desc_layout(lg_desc *desc, lg_layout layout, lg_size unit_align);

/// Returns true if a tensor is isotropic.
/// 
/// Tensors with a rank of zero, and all scalars are considered isotropic,
/// while all vectors are considered anisotropic.
static inline bool lg_desc_is_isotropic(lg_desc desc);

lg_status lg_add(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_sub(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_contract(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_hadamard(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);

lg_status lg_loss_mse(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_loss_cross_entropy(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);

lg_status lg_relu(lg_expr *expr, lg_tensor y, const lg_tensor in);
lg_status lg_stable_softmax(lg_expr *expr, const lg_tensor y, const lg_tensor in);
lg_status lg_sigmoid(lg_expr *expr, lg_tensor y, const lg_tensor in);
lg_status lg_ln(lg_expr *expr, lg_tensor y, const lg_tensor in);

lg_status lg_backward(lg_expr *expr);
lg_tensor lg_transpose(lg_expr *expr, const lg_tensor in);

#endif // LG_CORE_H_


#ifdef LG_CORE_IMPLEMENTATION
#undef LG_CORE_IMPLEMENTATION

#include <libgrad/internal/debug.h>

static inline lg_size lg_desc_size_bytes(lg_desc desc) {
    if (desc.rank == 0) {
        return 0;
    }

    lg_size max_offset = 0;
    for (lg_size i = 0; i < desc.rank; i++) {
        if (desc.dim[i] > 0) {
            max_offset += (desc.dim[i] - 1) * desc.strides[i];
        }
    }

    return (max_offset + 1) * sizeof(lg_dtype);
}

void lg_copy_vector(lg_desc desc, lg_dtype *restrict dest, const lg_dtype *vector, lg_size copy_to_dim) {
    lg_size dim_offset = 0;
    for (lg_size i = 0; i < copy_to_dim; i++) {
        dim_offset += desc.dim[i];
    }
    const lg_size n_values = desc.dim[copy_to_dim];
    for (lg_size i = 0; i < n_values; i++) {
        dest[dim_offset + i] = vector[i];
    }
}

lg_status lg_desc_layout(lg_desc *desc, lg_layout layout, lg_size unit_align) {
#ifdef LG_SAFE
    if (desc->rank > LG_MAX_RANK) {
        return LG_STATUS_INVALID_RANK;
    }
#endif // LG_SAFE
    lg_size last_stride = 1;
    for (lg_size i = 1; i <= desc->rank; i++) {
        lg_size axis = layout == LG_LAYOUT_ROW_MAJOR ? desc->rank - i : i - 1;
        desc->strides[axis] = last_stride;
        last_stride *= desc->dim[desc->rank - i];
        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (unit_align > 1 && i == 1) {
            last_stride = (last_stride + unit_align - 1) & ~(unit_align - 1);
        }
    }

    return LG_STATUS_OK;
}

static inline bool lg_desc_is_isotropic(lg_desc desc) {
    switch (desc.rank) {
    // All vectors are anisotropic.
    case 2:
        return 0;
    // All scalars and tensors of rank zero are isotropic.
    case 1:
    case 0:
        return 1;
    default: {
        lg_size last_dim = desc.dim[0];
        for (lg_size i = 0; i < desc.rank; i++) {
            if (last_dim != desc.dim[i]) {
                return 0;
            }
        }
        return 1;
    }
    }
}

void lg_desc_tranpose(lg_desc *desc) {
    for (lg_size i = 0; i < desc->rank / 2; i++) {
        const lg_size opp = desc->rank - 1 - i;
        lg_size temp = desc->dim[i];
        desc->dim[i] = desc->dim[opp];
        desc->dim[opp] = temp;
    }
}

/// Returns the maximum rank of all of the descriptors
static inline lg_size __lg_desc_left_pad_dims(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank < max_rank) {
            lg_size shift = max_rank - descs[i]->rank;
            for (lg_size j = descs[i]->rank; j > 0; j--) {
                lg_size src_idx = j - 1;
                descs[i]->dim[src_idx + shift] = descs[i]->dim[src_idx];
                descs[i]->strides[src_idx + shift] = descs[i]->strides[src_idx];
            }
            for (lg_size j = 0; j < shift; j++) {
                descs[i]->dim[j] = 1;
                descs[i]->strides[j] = 0;
            }
            descs[i]->rank = max_rank;
        }
    }
    
    return max_rank;
}
    
lg_status lg_desc_broadcast(lg_desc **descs, lg_size n_descs) {
    const lg_size max_rank = __lg_desc_left_pad_dims(descs, n_descs);

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

    for (lg_size i = 0; i < n_descs; i++) {
        for (lg_size j = 0; j < max_rank; j++) {
            lg_size dim_current = (j < descs[i]->rank) ? descs[i]->dim[j] : 1;
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
    for (lg_size i = 0; i < n_descs; i++) {
        for (lg_size j = 1; j <= max_rank; j++) {
            if (descs[i]->dim[max_rank - j] < master_dim[max_rank - j]) {
                descs[i]->strides[max_rank - j] = 0;
                descs[i]->dim[max_rank - j] = master_dim[max_rank - j];
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

lg_status lg_desc_compute_contracted_dims(lg_desc *y, lg_desc *x0, lg_desc *x1, lg_size n_batch_axes) {
    // The logical tensor axes will be laid out as follows:
    // { [batch], [x0_free], [x1_free], [contracted] }
    //    reg      reg       reg       0          | y strides
    //    reg      reg       0         reg        | x0 strides
    //    reg      0         reg       reg        | x1 strides

    lg_desc y_cpy = *y;
    lg_desc x0_cpy = *x0;
    lg_desc x1_cpy = *x1;
    
    // x0.rank = n_batch + n_contracted + x0_free
    // x1.rank = n_batch + n_contracted + x1_free
    // y.rank = n_batch + x0_free + x1_free
    // ergo ...
    const lg_size n_contracted_axes = (x0->rank + x1->rank - y->rank - n_batch_axes) / 2;
    const lg_size x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const lg_size x1_first_free_axis = n_contracted_axes + n_batch_axes;

    // Batch axes are already in place
    lg_size r = n_batch_axes;

    // Free axes
    for (lg_size i = n_batch_axes; i < x0_first_contracted_axis; i++, r++) {
        x0->dim[r] = x0_cpy.dim[i];
        x0->strides[r] = x0_cpy.strides[i];
        x1->dim[r] = x0_cpy.dim[i];
        x1->strides[r] = 0;
        if (y->dim[r] != x0_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        y->strides[r] = y_cpy.strides[r];
    }
    for (lg_size i = x1_first_free_axis; i < x1_cpy.rank; i++, r++) {
        x0->dim[r] = x1_cpy.dim[i];
        x0->strides[r] = 0;
        x1->dim[r] = x1_cpy.dim[i];
        x1->strides[r] = x1_cpy.strides[i];
        if (y->dim[r] != x1_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        y->strides[r] = y_cpy.strides[r];
    }

    // Contracted axes
    if (n_contracted_axes > 0) {
        for (
            lg_size x0_ax = x0_first_contracted_axis, x1_ax = x1_first_free_axis - 1;
            x0_ax < x0_cpy.rank; // x1_ax > 0
            x0_ax++, x1_ax--, r++
        ) {
            x0->dim[r] = x0_cpy.dim[x0_ax];
            x0->strides[r] = x0_cpy.strides[x0_ax];
            x1->dim[r] = x0_cpy.dim[x0_ax];
            x1->strides[r] = x1_cpy.strides[x1_ax];
            y->dim[r] = x0_cpy.dim[x0_ax];
            y->strides[r] = 0;
        }
    }

    y->rank = r;
    x0->rank = r;
    x1->rank = r;

    return LG_STATUS_OK;
}

lg_status lg_desc_sort_dims(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (lg_size i = 0; i < max_rank; i++) {
        bool swapped = 0;
        for (lg_size j = 1; j < max_rank - i; j++) {
            const lg_size prev_dim = descs[0]->strides[j - 1];
            const lg_size cur_dim = descs[0]->strides[j];
            if (prev_dim < cur_dim) {
                // Swap the dimensions and strides for all of the tensors
                for (lg_size k = 0; k < n_descs; k++) {
                    lg_size temp = descs[k]->dim[j - 1];
                    descs[k]->dim[j - 1] = descs[k]->dim[j];
                    descs[k]->dim[j] = temp;
                    temp = descs[k]->strides[j - 1];
                    descs[k]->strides[j - 1] = descs[k]->strides[j];
                    descs[k]->strides[j] = temp;
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

lg_status lg_desc_coalesce_dims(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
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
            for (lg_size j = 0; j < n_descs; j++) {
                const lg_size d0 = descs[j]->dim[i];
                const lg_size d1 = descs[j]->dim[i + 1];
                const lg_size s0 = descs[j]->strides[i];
                const lg_size s1 = descs[j]->strides[i + 1];

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

            const lg_size new_dim = descs[0]->dim[i] * descs[0]->dim[i + 1];
            max_rank--;
            for (lg_size j = 0; j < n_descs; j++) {
                descs[j]->rank = max_rank;
                descs[j]->dim[i] = new_dim;
                if (descs[j]->strides[i + 1] != 0) {
                    descs[j]->strides[i] = descs[j]->strides[i + 1];
                }
                for (lg_size k = i + 1; k < max_rank; k++) {
                    descs[j]->dim[k] = descs[j]->dim[k + 1];
                    descs[j]->strides[k] = descs[j]->strides[k + 1];
                }
            }
        }
    }

    return LG_STATUS_OK;
}

static inline lg_status lg_expr_append(
    lg_expr *expr,
    const lg_opcode opcode,
    const lg_tensor x0,
    const lg_tensor x1,
    const lg_tensor y
) {
#ifdef LG_SAFE
    if (expr->len >= expr->cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
#endif // LG_SAFE

    lg_size next_idx = expr->len;
    expr->len += 1;
    expr->opcodes[next_idx] = opcode;
    expr->x0[next_idx] = x0;
    expr->x1[next_idx] = x1;
    expr->y[next_idx] = y;

    return LG_STATUS_OK;
}

bool lg_nditer_increment(lg_nditer *iter, lg_size axis) {
    const lg_size rank = iter->descs[0].rank;
    const lg_size first_tracked_dim = rank - iter->n_tracked_dims;
    const lg_size *restrict dim = iter->descs[0].dim;

    if (rank == 0) {
        return false;
    }

    axis += 1;
    while (axis > first_tracked_dim) {
        axis--;
        iter->coords[axis]++;
        if (iter->coords[axis] < dim[axis]) {
            for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
                iter->indices[i] += iter->descs[i].strides[axis];
            }
            return true; 
        }
        iter->coords[axis] = 0;
        for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
            iter->indices[i] -= iter->descs[i].strides[axis] * (dim[axis] - 1);
        }
    }

    return false;
}

void lg_nditer_goto(lg_nditer *iter, lg_size *coords) {
    for(lg_size i = 0; i < iter->n_tracked_dims; i++) {
        iter->coords[i] = coords[i];
    }

    for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
        iter->indices[i] = 0;
        for (lg_size j = 0; j < iter->n_tracked_dims; j++) {
            iter->indices[i] += iter->descs[i].strides[j] * coords[j];
        }
    }
}

lg_status lg_add(
    lg_expr *expr,
    lg_tensor y,
    const lg_tensor x0,
    const lg_tensor x1
) {
    lg_status status;
    status = lg_expr_append(expr, LG_OPCODE_ADD, x0, x1, y);
    if (status != LG_STATUS_OK) {
        return status;
    }
    const lg_size i = expr->len - 1;
    status = lg_desc_broadcast((lg_desc*[]){
        &expr->y[i].desc,
        &expr->x0[i].desc,
        &expr->x1[i].desc,
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = lg_desc_sort_dims((lg_desc*[]){
        &expr->y[i].desc,
        &expr->x0[i].desc,
        &expr->x1[i].desc,
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = lg_desc_coalesce_dims((lg_desc*[]){
        &expr->y[i].desc,
        &expr->x0[i].desc,
        &expr->x1[i].desc,
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }
    return LG_STATUS_OK;
}

#endif // LG_CORE_IMPLEMENTATION
