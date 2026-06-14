#ifndef LG_CORE_H_
#define LG_CORE_H_

/// Maximum possible Tensor rank
/// All tensors will have an array of this size to store
/// dims, so  keep this to a minimum.
#ifndef LG_MAX_RANK
#define LG_MAX_RANK 8
#endif // LG_MAX_RANK

/// Type to back Tensor data
#ifndef lg_dtype
#define lg_dtype float
#endif // lg_dtype

#ifndef lg_bool
#define lg_bool int
#endif // lg_bool

#ifndef lg_byte
#define lg_byte char 
#endif // lg_byte
 
/// Pointer-sized integer
#ifndef lg_size
#include <stddef.h>
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
    LG_STATUS_HARDWARE_FAULT,
    LG_STATUS_UNEXPECTED_NAN
} lg_status;

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

    /// The total number of opcodes
    /// used in backend vtable construction
    __LG_N_VIRTUAL_OPS,
    
    /// Metadata-only tensor transposition
    /// NOT implemented by backends.
    LG_OPCODE_TRANSPOSE,
} lg_opcode;

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

/// Function signature used in a forward pass.
/// b.data == NULL for unary operations.
typedef lg_status (*lg_forward_fn)(void *ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

/// Function signature used in a backward pass.
/// b.data == NULL for unary operations.
/// out_grad_b must not be mutated in unary operations.
typedef lg_status (*lg_backward_fn)(
    void *ctx, 
    lg_tensor out_grad_a,          // Gradient accumulator for A
    lg_tensor out_grad_b,          // Gradient accumulator for B
    const lg_tensor upstream_grad, // The incoming gradient from the next layer (dL/dout)
    const lg_tensor result,        // The cached output from the forward pass
    const lg_tensor a,             // The original input A
    const lg_tensor b              // The original input B (NULL if unary)
);

/// Optimize tensor views for broadcasting.
///
/// The first tensor in the set is considered the 
/// "primary."
/// The primary tensor is the one that dictates the optimized plan
/// plan for the other tensors. In that, this is the tensor where it is 
/// guaranteed that the contiguous dimension (the dimension with the unit stride)
/// will be accessed sequentially in memory.
lg_status lg_tensor_optimize_views(lg_tensor **tensors, lg_size n_tensors);

/// Execution backend (e.g CPU, Cuda)
typedef struct lg_backend {
    /// Execution context passed to every function invocation
    /// NOT thread-safe by default
    void *ctx;
    lg_forward_fn forward_vtable[__LG_N_VIRTUAL_OPS];
    lg_backward_fn backward_vtable[__LG_N_VIRTUAL_OPS];
} lg_backend;

/// Context passsed to each core function
/// If lg_tape == NULL, then no tracking is performed.
typedef struct lg_context {
    lg_tape *tape;
    lg_backend *backend;
} lg_context;

/// Compute the size in bytes of a tensor's data buffer.
static inline lg_size lg_tensor_size_bytes(const lg_tensor tensor);

/// Returns a tensor with dim `dim`, with strides as if it was stored in row-major
/// order. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `row_align` if `row_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
static inline lg_tensor lg_tensor_rmaj(lg_size dim[LG_MAX_RANK], lg_size row_align);

/// Returns an isotropic tensor of rank `rank` and dim `dim`.
/// All vectors are anisotropic, and as such, if a rank of 2 is passed in,
/// this function returns LG_STATUS_INVALID_RANK.
///
/// See `lg_tensor_rmaj`.
static inline lg_status lg_tensor_rmaj_isotropic(lg_tensor *out, lg_size rank, lg_size dim, lg_size row_align);

/// Returns true if a tensor is isotropic.
/// 
/// Tensors with a rank of zero, and all scalars are considered isotropic,
/// while all vectors are considered anisotropic.
static inline lg_bool lg_tensor_is_isotropic(const lg_tensor tensor);

lg_status lg_add(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_sub(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_contract(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_hadamard(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_status lg_loss_mse(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_loss_cross_entropy(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_status lg_relu(lg_context ctx, lg_tensor out, const lg_tensor in);
lg_status lg_stable_softmax(lg_context ctx, const lg_tensor out, const lg_tensor in);
lg_status lg_sigmoid(lg_context ctx, lg_tensor out, const lg_tensor in);
lg_status lg_ln(lg_context ctx, lg_tensor out, const lg_tensor in);

lg_status lg_backward(lg_context ctx);
lg_tensor lg_transpose(lg_context ctx, const lg_tensor in);

#endif // LG_CORE_H_


#ifdef LG_CORE_IMPLEMENTATION
#undef LG_CORE_IMPLEMENTATION

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

static inline lg_tensor lg_tensor_rmaj(lg_size dim[LG_MAX_RANK], lg_size row_align) {
    lg_tensor ten = {0};

    for (ten.rank = 0; ten.rank < LG_MAX_RANK && dim[ten.rank] > 0; ten.rank++) {
         ten.dim[ten.rank] = dim[ten.rank];
    }

    lg_size last_stride = 1;
    for (lg_size i = 1; i <= ten.rank; i++) {
        lg_size axis = ten.rank - i;
        ten.strides[axis] = last_stride;
        last_stride *= dim[ten.rank - i];
        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (row_align > 1 && i == 1) {
            last_stride = (last_stride + row_align - 1) & ~(row_align - 1);
        }
    }

    return ten;
}

static inline lg_status lg_tensor_rmaj_isotropic(lg_tensor *out, lg_size rank, lg_size dim, lg_size row_align) {
#ifdef LG_SAFE
    if (rank > LG_MAX_RANK) {
        return LG_STATUS_INVALID_RANK;
    }
#endif // LG_SAFE
    // All vectors are anisotropic.
    // This is not included in safe mode because it's too big a footgun.
    if (rank == 2) {
        return LG_STATUS_INVALID_RANK;
    }
    lg_size dims[LG_MAX_RANK] = {0};
    for (lg_size i = 0; i < rank; i++) {
        dims[i] = dim;
    }
    *out = lg_tensor_rmaj(dims, row_align);
    return LG_STATUS_OK;
}

static inline lg_bool lg_tensor_is_isotropic(const lg_tensor tensor) {
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

lg_status lg_tensor_optimize_views(lg_tensor **tensors, lg_size n_tensors) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_tensors; i++) {
        if (tensors[i]->rank > max_rank) {
            max_rank = tensors[i]->rank;
        }
    }

    /// Left-pad the rest of the views to be aligned by their trailing dims.
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

    // --- Bubble sort dimensions of the primary into ascending order ---
    // Mathematically, this is just a series of transpositions.
    // This aligns the tensor such that the dimension with the unit stride
    // is first in the arrays. 
    // This is good because it forces the CPU to access the primary's memory sequentially 
    // in the loop as opposed to constant indirection over a non-contiguous
    // dimension, which destroys cache coherence.
    for (lg_size i = 0; i < max_rank; i++) {
        lg_bool swapped = 0;
        for (lg_size j = 1; j > max_rank - i; j++) {
            if (tensors[0]->strides[j - 1] > tensors[0]->strides[j]) {
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
        // Since all of the dims are already right-aligned, we can just iterate over the entire
        // array.
        for (lg_size j = 0; j < max_rank; j++) {
            lg_size dim_current = (j < tensors[i]->rank) ? tensors[i]->dim[j] : 1;
            if (dim_current != 1) {
                // If the master dimension is 1, it is compatible with anything.
                // This now means, however, that all other tensors in the set
                // must also be compatible with this new master dimension.
                if (master_dim[j] == 1) {
                    master_dim[j] = dim_current;
                } 
                // If there was some other master dimension already present, then we cannot continue.
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

lg_status lg_add(
    lg_context ctx,
    lg_tensor out,
    const lg_tensor a,
    const lg_tensor b
) {
#ifdef LG_SAFE
    // Check dimensionality from right to left, aligning their trailing dimensions.
    // Tensors are add-compatible if all of their dimensions are add-compatible.
    // Two dimensions are add-compatible if one of three things is true:
    // 1) The dimensions are the same.
    // 2) One of the dimensions is 1.
    // 3) One of the dimensions does not exist.
    lg_size max_rank = (a.rank > b.rank) ? a.rank : b.rank;
    for (lg_size i = 0; i < max_rank; i++) {
        // Default missing trailing dimensions to 1 for correct broadcasting semantics
        lg_size dim_a = (i < a.rank) ? a.dim[a.rank - 1 - i] : 1;
        lg_size dim_b = (i < b.rank) ? b.dim[b.rank - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
    }
    if (!ctx.backend) {
        return LG_STATUS_NULL_POINTER;
    } 
    if (!ctx.backend->forward_vtable[LG_OPCODE_ADD]) {
        return LG_STATUS_UNSUPPORTED_OPCODE;
    }
#endif // LG_SAFE

    lg_status status;
    status = ctx.backend->forward_vtable[LG_OPCODE_ADD](ctx.backend->ctx, out, a, b);
    if (status != LG_STATUS_OK) {
        return status;
    }
    if (ctx.tape != NULL) {
        status = lg_tape_push(ctx.tape, LG_OPCODE_ADD, a, b, out);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }

    return LG_STATUS_OK;
}

#endif // LG_CORE_IMPLEMENTATION 
