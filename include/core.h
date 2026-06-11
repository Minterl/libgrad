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
    /// Metadata-only tensor transposition
    LG_OPCODE_TRANSPOSE,
    /// Element-wise ReLU
    LG_OPCODE_RELU,
    /// Element-wise stable softmax
    LG_OPCODE_STABLE_SOFTMAX,
    /// Element-wise sigmoid
    LG_OPCODE_SIGMOID,
    /// Element-wise natural log
    LG_OPCODE_LN,

    /// The total number of opcodes
    /// Used in backend vtable construction
    __LG_N_OPCODES__
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
    lg_tensor *inputs_b LG_CHECK_BOUNDS_NULLABLE(len);
    lg_tensor *outputs  LG_CHECK_BOUNDS(len);
} lg_tape;

/// Function signature used in a forward pass.
/// b.data == NULL for unary operations.
typedef void (*lg_forward_fn)(void *ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

/// Function signature used in a backward pass.
/// b.data == NULL for unary operations.
/// out_grad_b must not be mutated in unary operations.
typedef void (*lg_backward_fn)(
    void *ctx, 
    lg_tensor out_grad_a,          // Gradient accumulator for A
    lg_tensor out_grad_b,          // Gradient accumulator for B
    const lg_tensor upstream_grad, // The incoming gradient from the next layer (dL/dout)
    const lg_tensor result,        // The cached output from the forward pass
    const lg_tensor a,             // The original input A
    const lg_tensor b              // The original input B (NULL if unary)
);

/// Execution backend (e.g CPU, Cuda)
typedef struct lg_backend {
    /// Execution context passed to every function invocation
    /// NOT thread-safe by default
    void *ctx;
    lg_forward_fn forward_vtable[__LG_N_OPCODES__]   LG_CHECK_BOUNDS(__LG_N_OPCODES__);
    lg_backward_fn backward_vtable[__LG_N_OPCODES__] LG_CHECK_BOUNDS(__LG_N_OPCODES__);
} lg_backend;

/// Context passsed to each core function
/// If lg_tape == NULL, then no tracking is performed.
typedef struct lg_context {
    lg_tape *lg_tape;
    lg_backend *lg_backend;
} lg_context;

lg_status lg_add(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_sub(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_contract(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_hadamard(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_status lg_loss_mse(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);
lg_status lg_loss_cross_entropy(lg_context ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

lg_tensor lg_transpose(lg_context ctx, const lg_tensor in);
lg_status lg_relu(lg_context ctx, lg_tensor out, const lg_tensor in);
lg_status lg_stable_softmax(lg_context ctx, const lg_tensor out, const lg_tensor in);
lg_status lg_sigmoid(lg_context ctx, lg_tensor out, const lg_tensor in);
lg_status lg_ln(lg_context ctx, lg_tensor out, const lg_tensor in);

#endif // LG_CORE_H_

#ifdef LG_CORE_IMPLEMENTATION_
#undef LG_CORE_IMPLEMENTATION_

#endif // LG_CORE_IMPLEMENTATION_ 
