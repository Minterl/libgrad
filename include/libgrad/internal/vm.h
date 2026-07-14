#ifndef LG_VM_H_
#define LG_VM_H_

#include <libgrad/internal/core.h>

/// Represents a single Tensor backed by `data`
///
/// Expressed in pure SSA form.
///
/// Generally, these are thin handles that should 
/// live on the stack and be shallow copied as necessary.
///
/// The size `data`, respectively, should product of the
/// non-zero values in `dim`.
///
/// Backing buffers are stored stride-major.
typedef struct lg_tensor {
    /// Since the exprs are pure SSA, the time the (y) value is born at
    /// functions as a globally unique identifier.
    uint32_t born_at;
    
    /// The shape descriptor of the tensor.
    lg_desc desc;
    
    /// The tensor's primary backing buffer.
    lg_scalar *data;
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
    LG_OPCODE_NOP,
    /// Element-wise ReLU
    LG_OPCODE_RELU,
    /// Element-wise stable softmax
    LG_OPCODE_STABLE_SOFTMAX,
    /// Element-wise sigmoid
    LG_OPCODE_SIGMOID,
    /// Element-wise natural log
    LG_OPCODE_LN,
} lg_opcode;

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

/// Ensure a specific tensor is present after finishing execution of an expr.
/// Mutates the data pointer in x such that it can be read from normally.
///
/// This must be called AFTER allocating memory for the expr.
lg_status lg_pin(lg_expr *expr, lg_tensor *x);

lg_status lg_nop(lg_expr *expr, lg_tensor x);

lg_status lg_add(lg_expr *expr, lg_tensor *y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_sub(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_contract(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_hadamard(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);

lg_status lg_loss_mse(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_loss_cross_entropy(lg_expr *expr, lg_tensor y, const lg_tensor x0, const lg_tensor x1);

lg_status lg_relu(lg_expr *expr, lg_tensor y, const lg_tensor in);
lg_status lg_stable_softmax(lg_expr *expr, const lg_tensor y, const lg_tensor in);
lg_status lg_sigmoid(lg_expr *expr, lg_tensor y, const lg_tensor in);
lg_status lg_ln(lg_expr *expr, lg_tensor y, const lg_tensor in);

lg_status lg_expr_compile(lg_expr *expr);

#endif // LG_VM_H_
